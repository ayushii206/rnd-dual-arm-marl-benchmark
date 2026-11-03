"""Cartesian action-disturbance robustness evaluation."""

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
import torch

from src.envs.dual_arm_transport_env import DualArmTransportEnv
from src.agents.ippo_networks import IPPOActorCritic
from src.agents.mappo_networks import MAPPOActor
from scripts.contact_stability_monitor import ContactStabilityMonitor
from scripts.diagnose_multiseed_contacts import CHECKPOINTS


TARGET = np.array([0.05, 0.0, 0.58], dtype=np.float64)
SUCCESS_DISTANCE = 0.01
DISTURBANCE_LEVELS = (0.0, 0.05, 0.10)
MAX_STEPS = 500

OUTPUT = Path("results/robustness/action_disturbance_5cm.csv")


def build_actors(algorithm, seed):
    if algorithm == "Classical":
        return None, None

    checkpoint_path = CHECKPOINTS[algorithm][seed]

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["algorithm"] == algorithm
    assert checkpoint["seed"] == seed
    assert checkpoint["action_mode"] == "cartesian"
    assert checkpoint["action_dim"] == 3
    assert checkpoint["total_steps"] == 10000
    assert np.allclose(checkpoint["target_position"], TARGET)
    assert np.isclose(
        checkpoint["success_distance"], SUCCESS_DISTANCE
    )

    actors = {}

    for name in ("left", "right"):
        if algorithm == "IPPO":
            actor = IPPOActorCritic(action_dim=3)
            actor.load_state_dict(checkpoint["models"][name])
        else:
            actor = MAPPOActor(action_dim=3)
            actor.load_state_dict(checkpoint["actors"][name])

        actor.eval()
        actors[name] = actor

    return actors, checkpoint_path


def evaluate(algorithm, seed, disturbance_std):
    torch.manual_seed(100)

    actors, checkpoint_path = build_actors(algorithm, seed)

    env = DualArmTransportEnv(
        target_position=TARGET,
        success_distance=SUCCESS_DISTANCE,
        action_mode=(
            "cartesian"
        ),
        progress_weight=100.0,
    )

    # Initialize the validated 2 kg bilateral grasp.
    observations = env.reset()
    body_id = env.model.body("transport_object").id
    assert np.isclose(env.model.body_mass[body_id], 2.0)

    disturbance_seed = 2026
    rng = np.random.default_rng(disturbance_seed)

    monitor = ContactStabilityMonitor(env)
    original_step = env.controller.step

    def monitored_step(left_q, right_q):
        original_step(left_q, right_q)
        monitor.record()

    env.controller.step = monitored_step

    initial_position = env.get_object_state()[0].copy()

    info = None
    terminated = False
    truncated = False
    rl_steps = 0

    for step in range(1, MAX_STEPS + 1):
        if algorithm == "Classical":
            position = env.get_object_state()[0].copy()
            error = env.target_position - position
            action = np.clip(10.0 * error, -0.5, 0.5)
            actions = {
                "left": action.copy(),
                "right": action.copy(),
            }
        else:
            actions = {}

            for name in ("left", "right"):
                observation = torch.as_tensor(
                    observations[name], dtype=torch.float32
                ).unsqueeze(0)

                with torch.no_grad():
                    action = torch.tanh(
                        actors[name].actor(observation)
                    )

                actions[name] = (
                    action.squeeze(0).cpu().numpy()
                )

        # Independent Gaussian action-command disturbances.
        disturbed_actions = {}
        for name in ("left", "right"):
            disturbance = rng.normal(
                0.0, disturbance_std, size=3
            )
            disturbed_actions[name] = np.clip(
                np.asarray(actions[name], dtype=np.float64)
                + disturbance,
                -1.0,
                1.0,
            )

        (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        ) = env.step(disturbed_actions)

        rl_steps = step

        if terminated or truncated:
            break

    summary = monitor.summary()
    final_position = env.get_object_state()[0]

    return {
        "algorithm": algorithm,
        "seed": seed if seed is not None else "NA",
        "checkpoint": checkpoint_path or "Classical",
        "mass_kg": 2.0,
        "disturbance_std": disturbance_std,
        "disturbance_seed": disturbance_seed,
        "disturbance_distribution": "Gaussian_action_xyz",
        "success": bool(info["success"]),
        "dropped": bool(info["dropped"]),
        "truncated": bool(truncated),
        "rl_steps": rl_steps,
        "physics_steps": summary["physics_steps"],
        "final_distance_mm": (
            1000 * float(info["distance_to_target"])
        ),
        "forward_displacement_mm": (
            1000 * float(
                final_position[0] - initial_position[0]
            )
        ),
        "mean_tilt_deg": summary["mean_tilt_deg"],
        "max_tilt_deg": summary["max_tilt_deg"],
        "max_orientation_deg": summary["max_orientation_deg"],
        "max_angular_speed_rad_s": (
            summary["max_angular_speed_rad_s"]
        ),
        "both_grasp_contact_pct": (
            summary["both_grasp_contact_pct"]
        ),
        "missing_left_steps": summary["missing_left_steps"],
        "missing_right_steps": summary["missing_right_steps"],
        "palm_object_steps": (
            summary["affected_steps"].get("palm_object", 0)
        ),
        "object_environment_steps": (
            summary["affected_steps"].get(
                "object_environment", 0
            )
        ),
        "inter_arm_contact_steps": (
            summary["affected_steps"].get("inter_arm_contact", 0)
        ),
        "robot_environment_contact_steps": (
            summary["affected_steps"].get(
                "robot_environment_contact", 0
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=("Classical", "IPPO", "MAPPO", "all"),
        default="all",
    )
    parser.add_argument(
        "--disturbance",
        type=float,
        choices=DISTURBANCE_LEVELS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=(42, 123, 456),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    args = parser.parse_args()

    torch.set_num_threads(1)

    algorithms = (
        ("Classical", "IPPO", "MAPPO")
        if args.algorithm == "all"
        else (args.algorithm,)
    )
    levels = (
        (args.disturbance,)
        if args.disturbance is not None
        else DISTURBANCE_LEVELS
    )

    experiments = []
    for disturbance_std in levels:
        for algorithm in algorithms:
            seeds = (
                (None,)
                if algorithm == "Classical"
                else (
                    (args.seed,)
                    if args.seed is not None
                    else (42, 123, 456)
                )
            )
            for seed in seeds:
                experiments.append(
                    (algorithm, seed, disturbance_std)
                )

    print(f"Planned evaluations: {len(experiments)}")
    for algorithm, seed, disturbance_std in experiments:
        print(
            f"  {algorithm} | seed={seed} | "
            f"disturbance_std={disturbance_std}"
        )

    if args.dry_run:
        print("Dry run complete. No simulations executed.")
        return

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {OUTPUT}"
        )

    rows = []
    for algorithm, seed, disturbance_std in experiments:
        print(
            f"\\nEvaluating {algorithm}, "
            f"seed={seed}, disturbance={disturbance_std}",
            flush=True,
        )
        row = evaluate(algorithm, seed, disturbance_std)
        rows.append(row)
        print(
            f"Success={row['success']} | "
            f"Steps={row['rl_steps']} | "
            f"Final distance={row['final_distance_mm']:.2f} mm",
            flush=True,
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\\nSaved {len(rows)} evaluations to {OUTPUT}")


if __name__ == "__main__":
    main()
