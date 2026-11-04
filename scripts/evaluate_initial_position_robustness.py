"""Evaluate forward initial-position perturbation robustness."""

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
PERTURBATION_COMMANDS = (0.0, 0.2, 0.5, 1.0)
MAX_STEPS = 500

OUTPUT = Path("results/initial_position/initial_position_robustness.csv")


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


def evaluate(algorithm, seed, perturbation_command):
    torch.manual_seed(100)

    actors, checkpoint_path = build_actors(algorithm, seed)

    evaluation_target = TARGET.copy()
    evaluation_target[0] = TARGET[0]

    env = DualArmTransportEnv(
        target_position=evaluation_target,
        success_distance=SUCCESS_DISTANCE,
        action_mode=(
            "cartesian"
        ),
        progress_weight=100.0,
    )

    # Perform the validated 2 kg grasp-and-lift initialization.
    observations = env.reset()

    # Keep the original 2 kg payload unchanged.
    body_id = env.model.body("transport_object").id
    assert np.isclose(env.model.body_mass[body_id], 2.0)

    # Verify the requested target is active.
    assert np.allclose(env.target_position, evaluation_target)


    # Preparation is separate from the transport evaluation.
    preparation_physics_steps = 0
    preparation_missing_contact_steps = 0

    if perturbation_command != 0.0:
        preparation_monitor = ContactStabilityMonitor(env)
        original_preparation_step = env.controller.step

        def preparation_step(left_q, right_q):
            original_preparation_step(left_q, right_q)
            preparation_monitor.record()

        env.controller.step = preparation_step

        try:
            forward = {
                "left": np.array([perturbation_command, 0.0, 0.0]),
                "right": np.array([perturbation_command, 0.0, 0.0]),
            }
            zero = {
                "left": np.zeros(3),
                "right": np.zeros(3),
            }

            for _ in range(10):
                env.apply_actions(forward)

            for _ in range(50):
                env.apply_actions(zero)
        finally:
            env.controller.step = original_preparation_step

        summary_preparation = preparation_monitor.summary()
        preparation_physics_steps = summary_preparation["physics_steps"]
        preparation_missing_contact_steps = (
            summary_preparation["missing_left_steps"]
            + summary_preparation["missing_right_steps"]
        )

        assert preparation_physics_steps == 600
        assert preparation_missing_contact_steps == 0

        position, quaternion, velocity, angular_velocity = (
            env.get_object_state()
        )
        left_contacts, right_contacts = env.get_grasp_contacts()

        assert left_contacts > 0 and right_contacts > 0
        assert np.linalg.norm(velocity) < 0.001
        assert np.linalg.norm(angular_velocity) < 0.001

        # apply_actions does not advance the episode counter.
        assert env._episode_step_count == 0

        # Update policy observations and reward reference.
        observations = env.get_observations()
        env._previous_target_distance = float(
            np.linalg.norm(env.get_transport_target_error())
        )

    assert env._episode_step_count == 0

    monitor = ContactStabilityMonitor(env)
    original_step = env.controller.step

    def monitored_step(left_q, right_q):
        original_step(left_q, right_q)
        monitor.record()

    env.controller.step = monitored_step

    initial_position = env.get_object_state()[0].copy()
    initial_distance_mm = 1000.0 * float(
        np.linalg.norm(env.target_position - initial_position)
    )


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
        "training_distance_cm": 5.0,
        "evaluation_distance_cm": 5.0,
        "perturbation_command": float(perturbation_command),
        "initial_x_m": float(initial_position[0]),
        "initial_y_m": float(initial_position[1]),
        "initial_z_m": float(initial_position[2]),
        "initial_distance_mm": initial_distance_mm,
        "preparation_physics_steps": preparation_physics_steps,
        "preparation_missing_contact_steps": preparation_missing_contact_steps,
        "evaluation_target_x_m": float(evaluation_target[0]),
        "mass_kg": 2.0,
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
        "--perturbation-command",
        type=float,
        choices=PERTURBATION_COMMANDS,
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

    perturbations = (
        (args.perturbation_command,)
        if args.perturbation_command is not None
        else PERTURBATION_COMMANDS
    )

    experiments = []

    for perturbation_command in perturbations:
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
                    (algorithm, seed, perturbation_command)
                )

    print(f"Planned evaluations: {len(experiments)}")

    for algorithm, seed, perturbation_command in experiments:
        print(
            f"  {algorithm} | seed={seed} | "
            f"command={perturbation_command:g}"
        )

    if args.dry_run:
        print("Dry run complete. No simulations executed.")
        return

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {OUTPUT}"
        )

    rows = []

    for algorithm, seed, perturbation_command in experiments:
        print(
            f"\\nEvaluating {algorithm}, seed={seed}, "
            f"command={perturbation_command:g}",
            flush=True,
        )

        row = evaluate(algorithm, seed, perturbation_command)
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
