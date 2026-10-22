from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.envs.dual_arm_transport_env import (
    DualArmTransportEnv,
    TRANSPORT_TARGET_POSITION,
)


def main():
    print("=" * 70)
    print("RL OBSERVATION TEST")
    print("=" * 70)

    env = DualArmTransportEnv()

    print()
    print("Resetting to validated lifted state...")

    env.reset()

    observations = env.get_observations()

    left_obs = observations["left"]
    right_obs = observations["right"]

    global_state = env.get_global_state()

    (
        object_position,
        object_quaternion,
        object_linear_velocity,
        object_angular_velocity,
    ) = env.get_object_state()

    target_error = (
        env.get_transport_target_error()
    )

    print()
    print("Dimensions:")
    print(
        "  left observation:",
        left_obs.shape,
    )
    print(
        "  right observation:",
        right_obs.shape,
    )
    print(
        "  MAPPO global state:",
        global_state.shape,
    )

    print()
    print("Task state:")
    print(
        "  object position:",
        object_position,
    )
    print(
        "  object quaternion [w x y z]:",
        object_quaternion,
    )
    print(
        "  object linear velocity:",
        object_linear_velocity,
    )
    print(
        "  object angular velocity:",
        object_angular_velocity,
    )
    print(
        "  transport target:",
        TRANSPORT_TARGET_POSITION,
    )
    print(
        "  target error:",
        target_error,
    )

    print()
    print("Observation sanity:")

    left_finite = bool(
        np.all(
            np.isfinite(left_obs)
        )
    )

    right_finite = bool(
        np.all(
            np.isfinite(right_obs)
        )
    )

    global_finite = bool(
        np.all(
            np.isfinite(global_state)
        )
    )

    dimensions_correct = (
        left_obs.shape == (33,)
        and right_obs.shape == (33,)
        and global_state.shape == (50,)
    )

    target_error_correct = bool(
        np.allclose(
            target_error,
            TRANSPORT_TARGET_POSITION
            - object_position,
            atol=1e-12,
        )
    )

    target_is_forward = (
        target_error[0] > 0.0
    )

    print(
        "  all local values finite:",
        left_finite and right_finite,
    )

    print(
        "  all global values finite:",
        global_finite,
    )

    print(
        "  dimensions correct:",
        dimensions_correct,
    )

    print(
        "  target-error calculation correct:",
        target_error_correct,
    )

    print(
        "  target lies in +X direction:",
        target_is_forward,
    )

    passed = all([
        left_finite,
        right_finite,
        global_finite,
        dimensions_correct,
        target_error_correct,
        target_is_forward,
    ])

    print()
    print("=" * 70)

    if passed:
        print("RL OBSERVATIONS: PASS")
    else:
        print("RL OBSERVATIONS: FAIL")
        raise RuntimeError(
            "RL observation validation failed."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
    