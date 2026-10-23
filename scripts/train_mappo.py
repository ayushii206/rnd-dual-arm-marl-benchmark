from pathlib import Path
import argparse
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.dual_arm_transport_env import DualArmTransportEnv
from src.agents.mappo_networks import (
    MAPPOActor,
    MAPPOCentralizedCritic,
)
from src.agents.mappo_buffer import MAPPORolloutBuffer
from src.agents.mappo_trainer import MAPPOTrainer

AGENTS = ("left", "right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-distance", type=float, default=0.05)
    parser.add_argument("--success-distance", type=float, default=0.01)
    parser.add_argument("--progress-weight", type=float, default=50.0)
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="mappo_smoke.pt",
    )
    args = parser.parse_args()

    if args.total_steps <= 0 or args.rollout_steps <= 0:
        parser.error("Step counts must be positive.")
    if not np.isfinite(args.target_distance) or not (
        0.0 < args.target_distance <= 0.38
    ):
        parser.error("Invalid target distance.")
    if not np.isfinite(args.success_distance) or (
        args.success_distance <= 0
    ):
        parser.error("Invalid success distance.")
    if (
        Path(args.checkpoint_name).name != args.checkpoint_name
        or not args.checkpoint_name.endswith(".pt")
    ):
        parser.error("Checkpoint name must be a .pt filename.")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = DualArmTransportEnv(
        target_position=np.array(
            [args.target_distance, 0.0, 0.58],
            dtype=np.float64,
        ),
        success_distance=args.success_distance,
        action_mode="cartesian",
        progress_weight=args.progress_weight,
    )

    actors = {
        name: MAPPOActor(action_dim=3)
        for name in AGENTS
    }
    critic = MAPPOCentralizedCritic()
    trainer = MAPPOTrainer(
        actors,
        critic,
        update_epochs=2,
        minibatch_size=32,
    )

    observations = env.reset()
    total_steps = 0
    completed_episodes = 0
    successful_episodes = 0
    episode_return = 0.0
    episode_length = 0
    update_number = 0

    print("=" * 70)
    print("MAPPO MUJOCO TRAINING")
    print("=" * 70)
    print("Total steps:", args.total_steps)
    print("Rollout steps:", args.rollout_steps)
    print("Seed:", args.seed)
    print("Action mode:", env.action_mode)
    print("Target:", env.target_position)
    print("Success tolerance:", env.success_distance)
    print("Progress reward weight:", env.progress_weight)

    while total_steps < args.total_steps:
        buffer = MAPPORolloutBuffer(action_dim=3)
        rollout_rewards = []

        steps_this_rollout = min(
            args.rollout_steps,
            args.total_steps - total_steps,
        )

        for _ in range(steps_this_rollout):
            state = env.get_global_state()
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                value = critic(state_tensor).item()

            actions = {}
            raw_actions = {}
            log_probs = {}

            for name in AGENTS:
                obs_tensor = torch.as_tensor(
                    observations[name],
                    dtype=torch.float32,
                ).unsqueeze(0)

                with torch.no_grad():
                    action, raw_action, log_prob = (
                        actors[name].sample_action(obs_tensor)
                    )

                actions[name] = (
                    action.squeeze(0).cpu().numpy()
                )
                raw_actions[name] = (
                    raw_action.squeeze(0).cpu().numpy()
                )
                log_probs[name] = log_prob.item()

            (
                next_observations,
                rewards,
                terminated,
                truncated,
                info,
            ) = env.step(actions)

            if not np.isclose(
                rewards["left"], rewards["right"]
            ):
                raise RuntimeError("Expected shared team reward.")

            next_state = env.get_global_state()
            with torch.no_grad():
                next_value = critic(
                    torch.as_tensor(
                        next_state,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                ).item()

            team_reward = float(rewards["left"])

            buffer.add(
                global_state=state,
                observations=observations,
                raw_actions=raw_actions,
                log_probs=log_probs,
                reward=team_reward,
                value=value,
                next_value=next_value,
                terminated=terminated,
                truncated=truncated,
            )

            rollout_rewards.append(team_reward)
            episode_return += team_reward
            episode_length += 1
            total_steps += 1

            if terminated or truncated:
                completed_episodes += 1
                successful_episodes += int(info["success"])

                print(
                    f"Episode {completed_episodes}: "
                    f"return={episode_return:.3f}, "
                    f"length={episode_length}, "
                    f"distance={info['distance_to_target']:.4f}m, "
                    f"success={info['success']}, "
                    f"dropped={info['dropped']}"
                )

                observations = env.reset()
                episode_return = 0.0
                episode_length = 0
            else:
                observations = next_observations

        update_number += 1
        metrics = trainer.update(buffer)

        print(
            f"Update {update_number} | "
            f"steps={total_steps} | "
            f"mean_reward={np.mean(rollout_rewards):.5f} | "
            f"policy_loss={metrics['policy_loss']:.5f} | "
            f"value_loss={metrics['value_loss']:.5f} | "
            f"approx_kl={metrics['approx_kl']:.5f}"
        )

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "mappo"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / args.checkpoint_name

    torch.save(
        {
            "algorithm": "MAPPO",
            "action_mode": "cartesian",
            "action_dim": 3,
            "target_position": env.target_position.tolist(),
            "success_distance": env.success_distance,
            "total_steps": total_steps,
            "rollout_steps": args.rollout_steps,
            "progress_weight": env.progress_weight,
            "seed": args.seed,
            "completed_episodes": completed_episodes,
            "successful_episodes": successful_episodes,
            "actors": {
                name: actors[name].state_dict()
                for name in AGENTS
            },
            "critic": critic.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
        },
        checkpoint_path,
    )

    print("=" * 70)
    print("MAPPO TRAINING COMPLETE")
    print("Environment steps:", total_steps)
    print("Completed episodes:", completed_episodes)
    if completed_episodes:
        print(
            "Training success rate:",
            f"{100 * successful_episodes / completed_episodes:.1f}%",
        )
    print("Checkpoint:", checkpoint_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
