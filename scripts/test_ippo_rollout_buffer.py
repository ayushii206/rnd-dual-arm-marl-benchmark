
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.rollout_buffer import RolloutBuffer


def make_buffer(rewards, terminated_last, truncated_last,
                last_next_value=0.0):
    buffer = RolloutBuffer(
        gamma=1.0,
        gae_lambda=1.0,
    )

    for i, reward in enumerate(rewards):
        is_last = i == len(rewards) - 1

        buffer.add(
            observation=np.zeros(33),
            raw_action=np.zeros(7),
            log_prob=-1.0,
            reward=reward,
            value=0.0,
            next_value=(
                last_next_value if is_last else 0.0
            ),
            terminated=(
                is_last and terminated_last
            ),
            truncated=(
                is_last and truncated_last
            ),
        )

    return buffer


def main():
    print("=" * 70)
    print("IPPO ROLLOUT BUFFER TEST")
    print("=" * 70)

    # A true terminal episode must not bootstrap.
    terminal_buffer = make_buffer(
        rewards=[1.0, 2.0, 3.0],
        terminated_last=True,
        truncated_last=False,
    )

    advantages, returns = (
        terminal_buffer.compute_advantages()
    )

    np.testing.assert_allclose(
        advantages, [6.0, 5.0, 3.0]
    )

    np.testing.assert_allclose(
        returns, [6.0, 5.0, 3.0]
    )

    print("Terminal GAE: PASS")

    # A time-limit truncation should bootstrap
    # from the final next-state value.
    truncated_buffer = make_buffer(
        rewards=[1.0, 2.0, 3.0],
        terminated_last=False,
        truncated_last=True,
        last_next_value=4.0,
    )

    advantages, _ = (
        truncated_buffer.compute_advantages()
    )

    np.testing.assert_allclose(
        advantages, [10.0, 9.0, 7.0]
    )

    print("Truncation bootstrap: PASS")

    # Also check a rollout that ends before
    # the environment episode has finished.
    partial_buffer = make_buffer(
        rewards=[1.0, 2.0, 3.0],
        terminated_last=False,
        truncated_last=False,
        last_next_value=4.0,
    )

    advantages, _ = (
        partial_buffer.compute_advantages()
    )

    np.testing.assert_allclose(
        advantages, [10.0, 9.0, 7.0]
    )

    print("Partial-rollout bootstrap: PASS")

    tensors = terminal_buffer.as_tensors()

    assert tensors["observations"].shape == (3, 33)
    assert tensors["raw_actions"].shape == (3, 7)
    assert tensors["advantages"].shape == (3,)
    assert tensors["returns"].shape == (3,)

    assert all(
        torch.isfinite(tensor).all()
        for tensor in tensors.values()
    )

    print("Tensor shapes and finite values: PASS")

    terminal_buffer.clear()
    assert len(terminal_buffer) == 0

    print("Buffer clearing: PASS")

    print("=" * 70)
    print("IPPO ROLLOUT BUFFER: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()

