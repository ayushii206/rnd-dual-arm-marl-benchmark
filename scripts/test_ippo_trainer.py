
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.ippo_networks import IPPOActorCritic
from src.agents.rollout_buffer import RolloutBuffer
from src.agents.ppo_trainer import PPOTrainer


def main():
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    agent = IPPOActorCritic()
    trainer = PPOTrainer(
        agent,
        update_epochs=2,
        minibatch_size=16,
    )

    buffer = RolloutBuffer()

    initial_parameters = [
        parameter.detach().clone()
        for parameter in agent.parameters()
    ]

    # Synthetic data tests the optimizer independently
    # of MuJoCo and environment reward design.
    for step in range(64):
        observation = rng.normal(
            size=33
        ).astype(np.float32)

        next_observation = rng.normal(
            size=33
        ).astype(np.float32)

        with torch.no_grad():
            obs_tensor = torch.from_numpy(
                observation
            ).unsqueeze(0)

            next_tensor = torch.from_numpy(
                next_observation
            ).unsqueeze(0)

            (
                action,
                raw_action,
                log_prob,
                value,
            ) = agent.sample_action(obs_tensor)

            next_value = agent.critic(
                next_tensor
            ).squeeze().item()

        buffer.add(
            observation=observation,
            raw_action=raw_action.squeeze(0).numpy(),
            log_prob=log_prob.item(),
            reward=float(rng.normal()),
            value=value.item(),
            next_value=next_value,
            terminated=(step == 63),
            truncated=False,
        )

    metrics = trainer.update(buffer)

    changed_parameters = any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            initial_parameters,
            agent.parameters(),
        )
    )

    assert changed_parameters, (
        "PPO did not update network parameters."
    )

    for key, value in metrics.items():
        assert np.isfinite(value), (
            f"Non-finite metric: {key}"
        )

    assert metrics["updates"] == 8

    print("=" * 70)
    print("IPPO TRAINER TEST")
    print("=" * 70)

    print("Rollout transitions:", len(buffer))
    print("Optimizer minibatch updates:", metrics["updates"])
    print("Network parameters changed:", changed_parameters)

    for key, value in metrics.items():
        if key != "updates":
            print(f"{key}: {value:.6f}")

    print("=" * 70)
    print("IPPO TRAINER: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()

