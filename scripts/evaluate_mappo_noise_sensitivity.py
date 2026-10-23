"""Evaluate MAPPO under different action-noise scales."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from src.agents.mappo_networks import MAPPOActor
from scripts.evaluate_ippo import evaluate_policy


AGENTS = ("left", "right")


class EvaluationActor(torch.nn.Module):
    """Adapt MAPPO actors to the shared evaluation function."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.actor = policy.actor

    def sample_action(self, observation):
        action, raw_action, log_prob = (
            self.policy.sample_action(observation)
        )
        return action, raw_action, log_prob, None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output",
        default=(
            "results/diagnostics/"
            "mappo_r4_noise_sensitivity.csv"
        ),
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("Episodes must be positive.")

    torch.set_num_threads(1)

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["algorithm"] != "MAPPO":
        raise ValueError("Expected MAPPO checkpoint.")

    if (
        checkpoint["action_mode"] != "cartesian"
        or checkpoint["action_dim"] != 3
    ):
        raise ValueError("Unexpected action configuration.")

    target_position = np.asarray(
        checkpoint["target_position"],
        dtype=np.float64,
    )

    scales = (0.0, 0.25, 0.5, 1.0, 1.5)

    all_results = []

    print("\nMAPPO R4 NOISE-SENSITIVITY EXPERIMENT")
    print("=" * 65)
    print("Checkpoint:", checkpoint_path)
    print("Episodes per scale:", args.episodes)
    print("Evaluation seed:", args.seed)

    for scale in scales:
        actors = {}

        for name in AGENTS:
            actor = MAPPOActor(
                action_dim=checkpoint["action_dim"]
            )

            actor.load_state_dict(
                checkpoint["actors"][name]
            )

            actor.eval()

            if scale > 0:
                with torch.no_grad():
                    actor.log_std.add_(math.log(scale))

            actors[name] = EvaluationActor(actor)

        stochastic = scale > 0

        print(f"\nEvaluating noise scale {scale:.2f}...")

        results = evaluate_policy(
            models=actors,
            target_position=target_position,
            success_distance=checkpoint["success_distance"],
            episodes=args.episodes,
            seed=args.seed,
            stochastic=stochastic,
            action_mode=checkpoint["action_mode"],
        )

        successes = sum(
            bool(row["success"]) for row in results
        )

        success_rate = (
            100.0 * successes / len(results)
        )

        mean_final_distance_mm = np.mean([
            float(row["final_distance_m"]) * 1000
            for row in results
        ])

        print(
            f"  Success: {successes}/{len(results)} "
            f"({success_rate:.1f}%)"
        )

        print(
            "  Mean final distance: "
            f"{mean_final_distance_mm:.2f} mm"
        )

        for row in results:
            record = dict(row)
            record["noise_scale"] = scale
            all_results.append(record)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("x", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETE")
    print("Total episodes:", len(all_results))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
