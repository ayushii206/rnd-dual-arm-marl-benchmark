"""Mass robustness evaluation for classical control, IPPO and MAPPO."""

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
MASSES = (2.0, 3.0, 4.0)
MAX_STEPS = 500

OUTPUT = Path("results/robustness/mass_robustness_5cm.csv")


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


def evaluate(algorithm, seed, mass):
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

    # Perform the validated 2 kg grasp-and-lift initialization.
    observations = env.reset()

    body_id = env.model.body("transport_object").id
    original_mass = float(env.model.body_mass[body_id])
    original_inertia = env.model.body_inertia[body_id].copy()

    assert np.isclose(original_mass, 2.0)

    # Modify payload dynamics only after grasp initialization.
    factor = mass / original_mass

    env.model.body_mass[body_id] = original_mass * factor
    env.model.body_inertia[body_id] = (
        original_inertia * factor
    )

    # Preserve the validated lifted grasp while updating
    # MuJoCo's model-dependent constants.
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state = np.empty(
        mujoco.mj_stateSize(env.model, state_spec),
        dtype=np.float64,
    )
    mujoco.mj_getState(
        env.model, env.data, state, state_spec
    )

    mujoco.mj_setConst(env.model, env.data)

    mujoco.mj_setState(
        env.model, env.data, state, state_spec
    )
    mujoco.mj_forward(env.model, env.data)

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
        "mass_kg": mass,
        "mass_multiplier": factor,
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
        "--mass",
        type=float,
        choices=MASSES,
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

    if args.algorithm == "all":
        algorithms = ("Classical", "IPPO", "MAPPO")
    else:
        algorithms = (args.algorithm,)

    masses = (args.mass,) if args.mass is not None else MASSES

    experiments = []

    for mass in masses:
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
                experiments.append((algorithm, seed, mass))

    print(f"Planned evaluations: {len(experiments)}")

    for algorithm, seed, mass in experiments:
        print(
            f"  {algorithm} | seed={seed} | mass={mass} kg"
        )

    if args.dry_run:
        print("Dry run complete. No simulations executed.")
        return

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {OUTPUT}"
        )

    rows = []

    for algorithm, seed, mass in experiments:
        print(
            f"\nEvaluating {algorithm}, "
            f"seed={seed}, mass={mass} kg",
            flush=True,
        )

        row = evaluate(algorithm, seed, mass)
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

    print(f"\nSaved {len(rows)} evaluations to {OUTPUT}")


if __name__ == "__main__":
    main()
