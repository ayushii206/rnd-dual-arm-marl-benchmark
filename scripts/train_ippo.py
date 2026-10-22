
from pathlib import Path
import argparse
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.dual_arm_transport_env import DualArmTransportEnv
from src.agents.ippo_networks import IPPOActorCritic
from src.agents.rollout_buffer import RolloutBuffer
from src.agents.ppo_trainer import PPOTrainer


AGENTS = ("left", "right")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--total-steps", type=int, default=128
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=64
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )

    args = parser.parse_args()

    if args.total_steps <= 0 or args.rollout_steps <= 0:
        parser.error("Step counts must be positive.")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = DualArmTransportEnv()

    models = {
        name: IPPOActorCritic()
        for name in AGENTS
    }

    trainers = {
        name: PPOTrainer(
            models[name],
            update_epochs=2,
            minibatch_size=32,
        )
        for name in AGENTS
    }

    observations = env.reset()

    total_steps = 0
    completed_episodes = 0
    episode_return = 0.0
    episode_length = 0
    update_number = 0

    print("=" * 70)
    print("IPPO MUJOCO TRAINING")
    print("=" * 70)
    print("Total steps:", args.total_steps)
    print("Rollout steps:", args.rollout_steps)
    print("Seed:", args.seed)

    while total_steps < args.total_steps:
        buffers = {
            name: RolloutBuffer()
            for name in AGENTS
        }

        steps_this_rollout = min(
            args.rollout_steps,
            args.total_steps - total_steps,
        )

        rollout_rewards = []
        last_info = {}

        for _ in range(steps_this_rollout):
            actions = {}
            samples = {}

            for name in AGENTS:
                observation = np.asarray(
                    observations[name],
                    dtype=np.float32,
                )

                obs_tensor = torch.from_numpy(
                    observation
                ).unsqueeze(0)

                with torch.no_grad():
                    (
                        action,
                        raw_action,
                        log_prob,
                        value,
                    ) = models[name].sample_action(
                        obs_tensor
                    )

                actions[name] = (
                    action.squeeze(0).numpy()
                )

                samples[name] = {
                    "observation": observation.copy(),
                    "raw_action": (
                        raw_action.squeeze(0).numpy()
                    ),
                    "log_prob": log_prob.item(),
                    "value": value.item(),
                }

            (
                next_observations,
                rewards,
                terminated,
                truncated,
                info,
            ) = env.step(actions)

            # Current environment uses shared episode flags.
            done = bool(terminated or truncated)

            for name in AGENTS:
                next_observation = np.asarray(
                    next_observations[name],
                    dtype=np.float32,
                )

                next_tensor = torch.from_numpy(
                    next_observation
                ).unsqueeze(0)

                with torch.no_grad():
                    next_value = (
                        models[name].critic(next_tensor)
                        .squeeze()
                        .item()
                    )

                buffers[name].add(
                    observation=(
                        samples[name]["observation"]
                    ),
                    raw_action=(
                        samples[name]["raw_action"]
                    ),
                    log_prob=(
                        samples[name]["log_prob"]
                    ),
                    reward=float(rewards[name]),
                    value=samples[name]["value"],
                    next_value=next_value,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )

            team_reward = float(rewards["left"])

            if not np.isclose(
                team_reward, float(rewards["right"])
            ):
                raise RuntimeError(
                    "Expected a shared team reward."
                )

            rollout_rewards.append(team_reward)
            episode_return += team_reward
            episode_length += 1
            total_steps += 1

            last_info = info

            if done:
                completed_episodes += 1

                print(
                    f"Episode {completed_episodes}: "
                    f"return={episode_return:.3f}, "
                    f"length={episode_length}, "
                    f"terminated={terminated}, "
                    f"truncated={truncated}"
                )

                observations = env.reset()
                episode_return = 0.0
                episode_length = 0
            else:
                observations = next_observations

        # Both agents update independently using
        # their own observations, actions and values.
        update_number += 1

        print(
            f"\nUpdate {update_number} | "
            f"environment steps: {total_steps}"
        )

        print(
            "  Mean team reward:",
            f"{np.mean(rollout_rewards):.6f}",
        )

        for name in AGENTS:
            metrics = trainers[name].update(
                buffers[name]
            )

            print(
                f"  {name}: "
                f"policy_loss="
                f"{metrics['policy_loss']:.5f}, "
                f"value_loss="
                f"{metrics['value_loss']:.5f}, "
                f"approx_kl="
                f"{metrics['approx_kl']:.5f}, "
                f"updates={metrics['updates']}"
            )

        print("  Last environment info:", last_info)

    checkpoint_dir = (
        PROJECT_ROOT / "checkpoints" / "ippo"
    )
    checkpoint_dir.mkdir(
        parents=True, exist_ok=True
    )

    checkpoint_path = (
        checkpoint_dir / "ippo_smoke_test.pt"
    )

    torch.save(
        {
            "algorithm": "IPPO",
            "seed": args.seed,
            "total_steps": total_steps,
            "update_number": update_number,
            "models": {
                name: models[name].state_dict()
                for name in AGENTS
            },
            "optimizers": {
                name: trainers[name].optimizer.state_dict()
                for name in AGENTS
            },
        },
        checkpoint_path,
    )

    print("\n" + "=" * 70)
    print("IPPO TRAINING SMOKE TEST COMPLETE")
    print("Environment steps:", total_steps)
    print("Completed episodes:", completed_episodes)
    print("Checkpoint:", checkpoint_path)
    print("=" * 70)


if __name__ == "__main__":
    main()

