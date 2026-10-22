from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.dual_arm_transport_env import (
    DualArmTransportEnv,
    MAX_EPISODE_STEPS,
)


def main():
    print("=" * 70)
    print("RL EPISODE TEST")
    print("=" * 70)

    env = DualArmTransportEnv()

    observations = env.reset()

    assert observations["left"].shape == (33,)
    assert observations["right"].shape == (33,)

    zero_actions = {
        "left": np.zeros(7),
        "right": np.zeros(7),
    }

    total_reward = 0.0

    terminated = False
    truncated = False

    for step_number in range(MAX_EPISODE_STEPS):
        (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        ) = env.step(zero_actions)

        assert np.isfinite(rewards["left"])
        assert rewards["left"] == rewards["right"]

        total_reward += rewards["left"]

        if terminated or truncated:
            break

    print()
    print("Episode steps:", step_number + 1)
    print("Simulation time [s]:", env.simulation_time)
    print("Total team reward:", total_reward)
    print("Final target distance [m]:",
          info["distance_to_target"])
    print("Success:", info["success"])
    print("Dropped:", info["dropped"])
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    assert terminated or truncated
    assert info["episode_step"] == step_number + 1

    print()
    print("=" * 70)
    print("RL EPISODE: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
