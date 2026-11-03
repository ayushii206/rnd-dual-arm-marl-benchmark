"""Sliding-friction robustness for classical control, IPPO and MAPPO."""

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
FRICTION_MULTIPLIERS = (1.0, 0.75, 0.5)
MAX_STEPS = 500

OUTPUT = Path("results/robustness/friction_robustness_5cm.csv")


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


def evaluate(algorithm, seed, friction_multiplier):
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

    # Initialize the original validated 2 kg bilateral grasp.
    observations = env.reset()

    body_id = env.model.body("transport_object").id
    assert np.isclose(env.model.body_mass[body_id], 2.0)

    # Scale sliding friction on finger and object grasp surfaces.
    # Leave torsional and rolling friction unchanged.
    finger_ids = list(range(68, 84)) + list(range(149, 165))
    grasp_ids = [166, 167, 168, 169]
    modified_ids = finger_ids + grasp_ids

    original_friction = env.model.geom_friction[
        modified_ids, 0
    ].copy()

    env.model.geom_friction[modified_ids, 0] = (
        original_friction * friction_multiplier
    )

    # Refresh contacts without resetting the lifted grasp.
    mujoco.mj_forward(env.model, env.data)

    expected_effective_friction = 1.5 * friction_multiplier
    matching_contacts = 0
    left_contacts = 0
    right_contacts = 0

    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)

        finger_id = None
        grasp_id = None

        if g1 in finger_ids and g2 in grasp_ids:
            finger_id, grasp_id = g1, g2
        elif g2 in finger_ids and g1 in grasp_ids:
            finger_id, grasp_id = g2, g1

        if finger_id is None:
            continue

        assert np.isclose(
            contact.friction[0],
            expected_effective_friction,
            atol=1e-5,
        ), (
            f"Unexpected effective friction: "
            f"{contact.friction[0]}"
        )

        matching_contacts += 1

        if finger_id < 84:
            left_contacts += 1
        else:
            right_contacts += 1

    assert left_contacts > 0 and right_contacts > 0, (
        "The initialized bilateral grasp was not preserved."
    )

    print(
        f"Verified friction={friction_multiplier:.2f}, "
        f"effective sliding friction="
        f"{expected_effective_friction:.4f}, "
        f"contact points={matching_contacts}",
        flush=True,
    )

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

        (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        ) = env.step(actions)

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
        "friction_multiplier": friction_multiplier,
        "effective_sliding_friction": (
            expected_effective_friction
        ),
        "initial_grasp_contact_points": matching_contacts,
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
        "--friction",
        type=float,
        choices=FRICTION_MULTIPLIERS,
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
    multipliers = (
        (args.friction,)
        if args.friction is not None
        else FRICTION_MULTIPLIERS
    )

    experiments = []
    for multiplier in multipliers:
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
                    (algorithm, seed, multiplier)
                )

    print(f"Planned evaluations: {len(experiments)}")
    for algorithm, seed, multiplier in experiments:
        print(
            f"  {algorithm} | seed={seed} | "
            f"friction={multiplier}"
        )

    if args.dry_run:
        print("Dry run complete. No simulations executed.")
        return

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {OUTPUT}"
        )

    rows = []
    for algorithm, seed, multiplier in experiments:
        print(
            f"\\nEvaluating {algorithm}, "
            f"seed={seed}, friction={multiplier}",
            flush=True,
        )
        row = evaluate(algorithm, seed, multiplier)
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
