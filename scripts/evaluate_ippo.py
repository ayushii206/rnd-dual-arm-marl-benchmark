
from pathlib import Path
import argparse
import csv
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.dual_arm_transport_env import DualArmTransportEnv
from src.agents.ippo_networks import IPPOActorCritic
from scripts.test_coordinated_transport import get_grasp_contacts

AGENTS = ("left", "right")


def build_models(checkpoint_path, seed):
    torch.manual_seed(seed)

    models = {
        name: IPPOActorCritic()
        for name in AGENTS
    }

    if checkpoint_path is not None:
        # Load only checkpoints you created or trust.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["algorithm"] != "IPPO":
            raise ValueError("Checkpoint is not IPPO.")

        for name in AGENTS:
            models[name].load_state_dict(
                checkpoint["models"][name]
            )

    for model in models.values():
        model.eval()

    return models


def evaluate_policy(
    models,
    target_position,
    success_distance,
    episodes,
    seed,
    stochastic,
):
    env = DualArmTransportEnv(
        target_position=target_position,
        success_distance=success_distance,
    )

    results = []

    for episode in range(episodes):
        observations = env.reset()
        initial_position = env.get_object_state()[0].copy()

        episode_return = 0.0
        episode_steps = 0
        left_missing_contact_steps = 0
        right_missing_contact_steps = 0
        both_missing_contact_steps = 0

        initial_left_contacts, initial_right_contacts = (
            get_grasp_contacts(env)
        )

        min_left_contacts = initial_left_contacts
        min_right_contacts = initial_right_contacts

        # Seeds action sampling for reproducible comparisons.
        torch.manual_seed(seed + episode)

        while True:
            actions = {}

            for name in AGENTS:
                obs_tensor = torch.as_tensor(
                    observations[name],
                    dtype=torch.float32,
                ).unsqueeze(0)

                with torch.no_grad():
                    if stochastic:
                        action, _, _, _ = (
                            models[name].sample_action(obs_tensor)
                        )
                    else:
                        # Deterministic evaluation uses the
                        # tanh of the Gaussian mean.
                        action = torch.tanh(
                            models[name].actor(obs_tensor)
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

            episode_return += float(rewards["left"])
            episode_steps += 1
            left_contacts, right_contacts = get_grasp_contacts(env)

            min_left_contacts = min(
                min_left_contacts, left_contacts
            )
            min_right_contacts = min(
                min_right_contacts, right_contacts
            )

            if left_contacts == 0:
                left_missing_contact_steps += 1

            if right_contacts == 0:
                right_missing_contact_steps += 1

            if left_contacts == 0 and right_contacts == 0:
                both_missing_contact_steps += 1

            if terminated or truncated:
                break

        final_position = env.get_object_state()[0].copy()

        results.append({
            "episode": episode + 1,
            "seed": seed + episode,
            "success": bool(info["success"]),
            "dropped": bool(info["dropped"]),
            "steps": episode_steps,
            "final_x_m": float(final_position[0]),
            "final_y_m": float(final_position[1]),
            "final_z_m": float(final_position[2]),
            "return": episode_return,
            "final_distance_m": float(
                info["distance_to_target"]
            ),
            "forward_displacement_m": float(
                final_position[0] - initial_position[0]
            ),
            "orientation_error_rad": float(
                info["orientation_error_rad"]
            ),
            "final_linear_speed_m_s": float(
                info["linear_speed"]
            ),
            "final_angular_speed_rad_s": float(
                info["angular_speed"]
            ),
            "truncated": bool(truncated),
            "initial_left_contacts": initial_left_contacts,
            "initial_right_contacts": initial_right_contacts,
            "final_left_contacts": left_contacts,
            "final_right_contacts": right_contacts,
            "min_left_contacts": min_left_contacts,
            "min_right_contacts": min_right_contacts,
            "left_missing_contact_steps": left_missing_contact_steps,
            "right_missing_contact_steps": right_missing_contact_steps,
            "both_missing_contact_steps": both_missing_contact_steps,
        })

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--target-distance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--success-distance",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ippo_evaluation.csv",
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("Episodes must be positive.")

    if (
        not np.isfinite(args.target_distance)
        or not 0.0 < args.target_distance <= 0.38
    ):
        parser.error("Invalid target distance.")

    if (
        not np.isfinite(args.success_distance)
        or args.success_distance <= 0.0
    ):
        parser.error("Invalid success distance.")

    torch.set_num_threads(1)

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else None
    )

    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            parser.error("Checkpoint file not found.")

        # Verify task configuration matches the checkpoint.
        metadata = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        expected_target = np.array([
            args.target_distance, 0.0, 0.58
        ])

        if not np.allclose(
            metadata["target_position"],
            expected_target,
        ):
            parser.error("Checkpoint target mismatch.")

        if not np.isclose(
            metadata["success_distance"],
            args.success_distance,
        ):
            parser.error("Checkpoint success tolerance mismatch.")

    models = build_models(
        checkpoint_path,
        args.seed,
    )

    target_position = np.array([
        args.target_distance, 0.0, 0.58
    ])

    results = evaluate_policy(
        models=models,
        target_position=target_position,
        success_distance=args.success_distance,
        episodes=args.episodes,
        seed=args.seed,
        stochastic=args.stochastic,
    )

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    success_rate = np.mean([
        row["success"] for row in results
    ]) * 100.0

    mean_distance = np.mean([
        row["final_distance_m"] for row in results
    ])

    mean_forward = np.mean([
        row["forward_displacement_m"] for row in results
    ])

    print("=" * 70)
    print("IPPO INDEPENDENT EVALUATION")
    print("=" * 70)
    print(
        "Policy:",
        checkpoint_path if checkpoint_path else "UNTRAINED"
    )
    print("Action mode:",
          "stochastic" if args.stochastic else "deterministic")
    print("Episodes:", args.episodes)
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Mean final distance: {mean_distance:.5f} m")
    print(f"Mean forward displacement: {mean_forward:.5f} m")
    print("CSV:", output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()


