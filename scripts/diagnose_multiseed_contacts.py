"""Deterministic multi-seed contact and stability evaluation."""

from pathlib import Path
import csv

import numpy as np
import torch

from src.envs.dual_arm_transport_env import DualArmTransportEnv
from src.agents.ippo_networks import IPPOActorCritic
from src.agents.mappo_networks import MAPPOActor
from scripts.contact_stability_monitor import ContactStabilityMonitor


CHECKPOINTS = {
    "IPPO": {
        42: "checkpoints/ippo/ippo_cartesian_r4_10k_5cm.pt",
        123: "checkpoints/ippo/ippo_r4_seed123_10k.pt",
        456: "checkpoints/ippo/ippo_r4_seed456_10k.pt",
    },
    "MAPPO": {
        42: "checkpoints/mappo/mappo_cartesian_r4_10k_5cm.pt",
        123: "checkpoints/mappo/mappo_r4_seed123_10k.pt",
        456: "checkpoints/mappo/mappo_r4_seed456_10k.pt",
    },
}

OUTPUT = Path(
    "results/comparison/multiseed_deterministic_5cm_stability.csv"
)


def evaluate(algorithm, seed, checkpoint_path):
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
    assert np.allclose(
        checkpoint["target_position"], [0.05, 0.0, 0.58]
    )

    torch.manual_seed(100)

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

    env = DualArmTransportEnv(
        target_position=np.asarray(
            checkpoint["target_position"], dtype=np.float64
        ),
        success_distance=checkpoint["success_distance"],
        action_mode="cartesian",
    )

    observations = env.reset()
    monitor = ContactStabilityMonitor(env)

    original_step = env.controller.step

    def monitored_step(left_q, right_q):
        original_step(left_q, right_q)
        monitor.record()

    env.controller.step = monitored_step

    info = None
    rl_steps = 0

    for rl_step in range(1, 501):
        actions = {}

        for name in ("left", "right"):
            observation = torch.as_tensor(
                observations[name], dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                action = torch.tanh(
                    actors[name].actor(observation)
                )

            actions[name] = action.squeeze(0).cpu().numpy()

        observations, rewards, terminated, truncated, info = (
            env.step(actions)
        )

        rl_steps = rl_step

        if terminated or truncated:
            break

    summary = monitor.summary()

    row = {
        "algorithm": algorithm,
        "seed": seed,
        "checkpoint": checkpoint_path,
        "success": bool(info["success"]),
        "dropped": bool(info["dropped"]),
        "rl_steps": rl_steps,
        "physics_steps": summary["physics_steps"],
        "final_distance_mm": (
            1000.0 * float(info["distance_to_target"])
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
        "truncated": bool(truncated),
    }

    categories = (
        "intentional_finger_object",
        "palm_object",
        "other_robot_object",
        "object_environment",
        "inter_arm_contact",
        "robot_environment_contact",
        "self_contact",
        "other_contact",
    )

    for category in categories:
        row[f"{category}_points"] = (
            summary["contact_points"].get(category, 0)
        )
        row[f"{category}_steps"] = (
            summary["affected_steps"].get(category, 0)
        )

    return row


def main():
    torch.set_num_threads(1)

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {OUTPUT}"
        )

    rows = []

    for algorithm, checkpoints in CHECKPOINTS.items():
        for seed, checkpoint_path in checkpoints.items():
            print(
                f"Evaluating {algorithm} seed {seed}...",
                flush=True,
            )

            row = evaluate(algorithm, seed, checkpoint_path)
            rows.append(row)

            print(
                f"  Success: {row['success']} | "
                f"Steps: {row['rl_steps']} | "
                f"Mean tilt: {row['mean_tilt_deg']:.4f} deg",
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
