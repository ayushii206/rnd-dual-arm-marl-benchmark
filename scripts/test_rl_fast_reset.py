from pathlib import Path
import sys
import time

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


def get_object_state(env):
    object_body_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        OBJECT_BODY_NAME,
    )

    if object_body_id < 0:
        raise RuntimeError(
            "Transport object body was not found."
        )

    position = (
        env.data.xpos[
            object_body_id
        ].copy()
    )

    linear_velocity = (
        env.data.cvel[
            object_body_id,
            3:6,
        ].copy()
    )

    angular_velocity = (
        env.data.cvel[
            object_body_id,
            0:3,
        ].copy()
    )

    return (
        position,
        linear_velocity,
        angular_velocity,
    )


def main():
    print("=" * 70)
    print("RL FAST RESET TEST")
    print("=" * 70)

    env = DualArmTransportEnv()

    # ========================================================
    # First reset:
    # build the validated state and cache it
    # ========================================================

    print()
    print("First reset: building and caching lifted state...")

    start = time.perf_counter()

    env.reset()

    first_reset_wall_time = (
        time.perf_counter() - start
    )

    (
        reference_position,
        reference_linear_velocity,
        reference_angular_velocity,
    ) = get_object_state(env)

    reference_qpos = env.data.qpos.copy()
    reference_qvel = env.data.qvel.copy()

    print(
        "  wall-clock time [s]:",
        first_reset_wall_time,
    )

    print(
        "  simulation time [s]:",
        env.simulation_time,
    )

    print(
        "  object position [m]:",
        reference_position,
    )

    # ========================================================
    # Disturb the simulation
    # ========================================================

    print()
    print("Disturbing simulation state...")

    env.data.qpos[0] += 0.1
    env.data.qvel[0] += 0.5

    mujoco.mj_forward(
        env.model,
        env.data,
    )

    disturbed_qpos_difference = float(
        np.max(
            np.abs(
                env.data.qpos
                - reference_qpos
            )
        )
    )

    print(
        "  maximum qpos disturbance:",
        disturbed_qpos_difference,
    )

    # ========================================================
    # Second reset:
    # should restore cache directly
    # ========================================================

    print()
    print("Second reset: restoring cached state...")

    start = time.perf_counter()

    env.reset()

    second_reset_wall_time = (
        time.perf_counter() - start
    )

    (
        restored_position,
        restored_linear_velocity,
        restored_angular_velocity,
    ) = get_object_state(env)

    qpos_error = float(
        np.max(
            np.abs(
                env.data.qpos
                - reference_qpos
            )
        )
    )

    qvel_error = float(
        np.max(
            np.abs(
                env.data.qvel
                - reference_qvel
            )
        )
    )

    position_error = float(
        np.linalg.norm(
            restored_position
            - reference_position
        )
    )

    linear_velocity_error = float(
        np.linalg.norm(
            restored_linear_velocity
            - reference_linear_velocity
        )
    )

    angular_velocity_error = float(
        np.linalg.norm(
            restored_angular_velocity
            - reference_angular_velocity
        )
    )

    print(
        "  wall-clock time [s]:",
        second_reset_wall_time,
    )

    print(
        "  simulation time [s]:",
        env.simulation_time,
    )

    print(
        "  qpos restoration error:",
        qpos_error,
    )

    print(
        "  qvel restoration error:",
        qvel_error,
    )

    print(
        "  object position restoration error [m]:",
        position_error,
    )

    print(
        "  object linear-velocity restoration error:",
        linear_velocity_error,
    )

    print(
        "  object angular-velocity restoration error:",
        angular_velocity_error,
    )

    # ========================================================
    # Acceptance
    # ========================================================

    state_restored = all([
        qpos_error <= 1e-12,
        qvel_error <= 1e-12,
        position_error <= 1e-12,
        linear_velocity_error <= 1e-12,
        angular_velocity_error <= 1e-12,
    ])

    episode_time_reset = (
        abs(env.simulation_time)
        <= 1e-12
    )

    fast_reset_is_faster = (
        second_reset_wall_time
        < first_reset_wall_time
    )

    print()
    print("Acceptance checks:")

    print(
        "  state restored exactly:",
        state_restored,
    )

    print(
        "  episode simulation time reset to zero:",
        episode_time_reset,
    )

    print(
        "  cached reset faster than first reset:",
        fast_reset_is_faster,
    )

    passed = all([
        state_restored,
        episode_time_reset,
        fast_reset_is_faster,
    ])

    print()
    print("=" * 70)

    if passed:
        print("RL FAST RESET: PASS")
    else:
        print("RL FAST RESET: FAIL")
        raise RuntimeError(
            "Fast RL reset validation failed."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()