from pathlib import Path
import argparse
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.mappo_networks import MAPPOActor
from scripts.evaluate_ippo import evaluate_policy

AGENTS = ("left", "right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--output",
        default="results/mappo_evaluation.csv",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("Episodes must be positive.")

    torch.set_num_threads(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error("Checkpoint file not found.")

    # Load only checkpoints you created or trust.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("algorithm") != "MAPPO":
        raise ValueError("Checkpoint is not MAPPO.")

    action_mode = checkpoint["action_mode"]
    action_dim = checkpoint["action_dim"]

    if action_mode != "cartesian" or action_dim != 3:
        raise ValueError("Unexpected MAPPO action configuration.")

    actors = {
        name: MAPPOActor(action_dim=action_dim)
        for name in AGENTS
    }

    for name in AGENTS:
        actors[name].load_state_dict(
            checkpoint["actors"][name]
        )
        actors[name].eval()

    target_position = np.asarray(
        checkpoint["target_position"],
        dtype=np.float64,
    )

    # evaluate_policy expects actor.sample_action() to
    # return four outputs. MAPPOActor returns three,
    # so adapt it without changing either network.
    class ActorAdapter:
        def __init__(self, actor):
            self.actor = actor

        def sample_action(self, observation):
            action, raw_action, log_prob = (
                self.actor.sample_action(observation)
            )
            return action, raw_action, log_prob, None

        def actor(self, observation):
            return self.actor.actor(observation)

    # Use the existing actor directly for deterministic
    # evaluation through a small wrapper module.
    class EvaluationActor(torch.nn.Module):
        def __init__(self, mappo_actor):
            super().__init__()
            self.policy = mappo_actor
            self.actor = mappo_actor.actor

        def sample_action(self, observation):
            action, raw_action, log_prob = (
                self.policy.sample_action(observation)
            )
            return action, raw_action, log_prob, None

    models = {
        name: EvaluationActor(actors[name])
        for name in AGENTS
    }

    results = evaluate_policy(
        models=models,
        target_position=target_position,
        success_distance=checkpoint["success_distance"],
        episodes=args.episodes,
        seed=args.seed,
        stochastic=args.stochastic,
        action_mode=action_mode,
    )

    import csv

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    success_rate = 100 * np.mean([
        row["success"] for row in results
    ])
    mean_distance = np.mean([
        row["final_distance_m"] for row in results
    ])
    mean_forward = np.mean([
        row["forward_displacement_m"] for row in results
    ])

    print("=" * 70)
    print("MAPPO INDEPENDENT EVALUATION")
    print("=" * 70)
    print("Checkpoint:", checkpoint_path)
    print(
        "Action selection:",
        "stochastic" if args.stochastic else "deterministic",
    )
    print("Episodes:", args.episodes)
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Mean final distance: {mean_distance:.5f} m")
    print(f"Mean forward displacement: {mean_forward:.5f} m")
    print("CSV:", output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
