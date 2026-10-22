
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.dual_arm_transport_env import DualArmTransportEnv


def get_tracking_state(env):
    object_pos = env.get_object_state()[0].copy()

    left_ee = env.data.site_xpos[
        env.left_ee_site_id
    ].copy()

    right_ee = env.data.site_xpos[
        env.right_ee_site_id
    ].copy()

    return {
        "object": object_pos,
        "left_relative": object_pos - left_ee,
        "right_relative": object_pos - right_ee,
    }


def run_test(env, name, action_fn, steps=100):
    env.reset()
    initial = get_tracking_state(env)

    max_left_drift = 0.0
    max_right_drift = 0.0

    for i in range(steps):
        _, _, terminated, truncated, _ = env.step(
            action_fn(i)
        )

        current = get_tracking_state(env)

        left_drift = np.linalg.norm(
            current["left_relative"]
            - initial["left_relative"]
        )

        right_drift = np.linalg.norm(
            current["right_relative"]
            - initial["right_relative"]
        )

        max_left_drift = max(
            max_left_drift, left_drift
        )

        max_right_drift = max(
            max_right_drift, right_drift
        )

        if terminated or truncated:
            break

    final = get_tracking_state(env)

    print(f"\n{name}")
    print(
        "  Object X displacement [m]:",
        final["object"][0] - initial["object"][0]
    )
    print(
        "  Max left relative drift [m]:",
        max_left_drift
    )
    print(
        "  Max right relative drift [m]:",
        max_right_drift
    )


def main():
    env = DualArmTransportEnv()
    rng = np.random.default_rng(42)

    run_test(
        env,
        "ZERO ACTIONS",
        lambda _: {
            "left": np.zeros(7),
            "right": np.zeros(7),
        },
    )

    run_test(
        env,
        "SMALL RANDOM ACTIONS",
        lambda _: {
            "left": rng.uniform(-0.2, 0.2, 7),
            "right": rng.uniform(-0.2, 0.2, 7),
        },
    )

    run_test(
        env,
        "DIRECTED ACTIONS",
        lambda _: {
            "left": np.array([0.2, 0, 0, 0, 0, 0, 0]),
            "right": np.array([0.2, 0, 0, 0, 0, 0, 0]),
        },
    )

    print("\nGRASP TRACKING DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()

