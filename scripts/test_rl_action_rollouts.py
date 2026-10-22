
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.dual_arm_transport_env import DualArmTransportEnv


def run_rollout(env, name, action_fn, steps=100):
    env.reset()

    initial_position = env.get_object_state()[0].copy()
    initial_distance = np.linalg.norm(
        env.get_transport_target_error()
    )

    final_info = None
    executed = 0

    for i in range(steps):
        actions = action_fn(i)

        observations, rewards, terminated, truncated, info = (
            env.step(actions)
        )

        assert all(
            np.all(np.isfinite(obs))
            for obs in observations.values()
        )
        assert np.isfinite(rewards["left"])

        final_info = info
        executed += 1

        if terminated or truncated:
            break

    final_position = env.get_object_state()[0].copy()
    final_distance = np.linalg.norm(
        env.get_transport_target_error()
    )

    print(f"\n{name}")
    print("  Steps:", executed)
    print("  Initial position:", initial_position)
    print("  Final position:", final_position)
    print("  Object displacement [m]:",
          np.linalg.norm(final_position - initial_position))
    print("  X displacement [m]:",
          final_position[0] - initial_position[0])
    print("  Distance improvement [m]:",
          initial_distance - final_distance)
    print("  Dropped:", final_info["dropped"])
    print("  Success:", final_info["success"])

    return final_position


def main():
    env = DualArmTransportEnv()
    rng = np.random.default_rng(42)

    zeros = lambda _: {
        "left": np.zeros(7),
        "right": np.zeros(7),
    }

    random_actions = lambda _: {
        "left": rng.uniform(-0.2, 0.2, size=7),
        "right": rng.uniform(-0.2, 0.2, size=7),
    }

    # Diagnostic only: joint 1 positive for both arms.
    # This is not guaranteed to move the object in +X.
    directed = lambda _: {
        "left": np.array([0.2, 0, 0, 0, 0, 0, 0]),
        "right": np.array([0.2, 0, 0, 0, 0, 0, 0]),
    }

    print("=" * 70)
    print("RL ACTION ROLLOUT COMPARISON")
    print("=" * 70)

    run_rollout(env, "ZERO ACTIONS", zeros)
    run_rollout(env, "SMALL RANDOM ACTIONS", random_actions)
    run_rollout(env, "DIRECTED JOINT ACTIONS", directed)

    print("\nRL ACTION ROLLOUTS: COMPLETED")


if __name__ == "__main__":
    main()

