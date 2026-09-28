from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.envs.dual_arm_transport_env import (
    DualArmTransportEnv,
)


OBJECT_BODY_NAME = "transport_object"


def main():
    print("=" * 70)
    print("RL LIFTED RESET TEST")
    print("=" * 70)

    env = DualArmTransportEnv()

    print()
    print("Constructing validated lifted state...")

    trajectories = env.reset_to_lifted_state()

    print("Lifted state construction complete.")

    object_body_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        OBJECT_BODY_NAME,
    )

    if object_body_id < 0:
        raise RuntimeError(
            "Transport object body was not found."
        )

    object_position = (
        env.data.xpos[
            object_body_id
        ].copy()
    )

    object_linear_velocity = (
        env.data.cvel[
            object_body_id,
            3:6,
        ].copy()
    )

    object_angular_velocity = (
        env.data.cvel[
            object_body_id,
            0:3,
        ].copy()
    )

    linear_speed = float(
        np.linalg.norm(
            object_linear_velocity
        )
    )

    angular_speed = float(
        np.linalg.norm(
            object_angular_velocity
        )
    )

    left_q = (
        env.controller.get_left_arm_qpos()
    )

    right_q = (
        env.controller.get_right_arm_qpos()
    )

    left_lift_error = float(
        np.max(
            np.abs(
                left_q
                - trajectories["left_lift_q"]
            )
        )
    )

    right_lift_error = float(
        np.max(
            np.abs(
                right_q
                - trajectories["right_lift_q"]
            )
        )
    )

    print()
    print("Lifted state:")
    print(
        "  simulation time [s]:",
        env.simulation_time,
    )
    print(
        "  object position [m]:",
        object_position,
    )
    print(
        "  object linear speed [m/s]:",
        linear_speed,
    )
    print(
        "  object angular speed [rad/s]:",
        angular_speed,
    )
    print(
        "  left max joint tracking error [rad]:",
        left_lift_error,
    )
    print(
        "  right max joint tracking error [rad]:",
        right_lift_error,
    )

    # --------------------------------------------------------
    # Basic lifted-state acceptance criteria
    # --------------------------------------------------------

    object_is_elevated = (
        object_position[2] > 0.565
    )

    object_is_slow = (
        linear_speed <= 0.01
    )

    object_rotation_is_slow = (
        angular_speed <= 0.10
    )

    arms_are_tracking = (
        left_lift_error <= 0.01
        and right_lift_error <= 0.01
    )

    print()
    print("Acceptance checks:")
    print(
        "  object elevated:",
        object_is_elevated,
    )
    print(
        "  object linear speed <= 0.01 m/s:",
        object_is_slow,
    )
    print(
        "  object angular speed <= 0.10 rad/s:",
        object_rotation_is_slow,
    )
    print(
        "  arm tracking error <= 0.01 rad:",
        arms_are_tracking,
    )

    passed = all([
        object_is_elevated,
        object_is_slow,
        object_rotation_is_slow,
        arms_are_tracking,
    ])

    print()
    print("=" * 70)

    if passed:
        print("RL LIFTED RESET: PASS")
    else:
        print("RL LIFTED RESET: FAIL")
        raise RuntimeError(
            "Validated lifted RL reset failed."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
    