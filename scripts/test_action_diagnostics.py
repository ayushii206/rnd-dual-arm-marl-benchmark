
import numpy as np

from src.envs.dual_arm_transport_env import DualArmTransportEnv


def run_case(name, action_fn, steps=100):
    env = DualArmTransportEnv(
        target_position=np.array([0.05, 0.0, 0.58]),
        success_distance=0.01,
    )

    env.reset()

    initial_position = env.get_object_state()[0].copy()

    missing_left = 0
    missing_right = 0
    completed_steps = 0
    final_info = {}

    for step in range(steps):
        actions = action_fn(step)

        _, _, terminated, truncated, info = env.step(actions)

        completed_steps += 1
        missing_left += int(not info["left_grasp_contact"])
        missing_right += int(not info["right_grasp_contact"])
        final_info = info

        if terminated or truncated:
            break

    final_position = env.get_object_state()[0].copy()
    displacement = final_position - initial_position

    print(f"\n{name}")
    print("-" * 55)
    print("Steps:", completed_steps)
    print("Object displacement [m]:", displacement)
    print("Final target distance [m]:",
          final_info["distance_to_target"])
    print("Final orientation error [rad]:",
          final_info["orientation_error_rad"])
    print("Missing left-contact steps:", missing_left)
    print("Missing right-contact steps:", missing_right)
    print("Final cooperative grasp:",
          final_info["cooperative_grasp"])
    print("Success:", final_info["success"])


def zero_actions(step):
    return {
        "left": np.zeros(7),
        "right": np.zeros(7),
    }


def constant_actions(step):
    return {
        "left": np.full(7, 0.25),
        "right": np.full(7, 0.25),
    }


def alternating_actions(step):
    sign = 1.0 if step % 2 == 0 else -1.0

    return {
        "left": np.full(7, 0.25 * sign),
        "right": np.full(7, 0.25 * sign),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("JOINT ACTION DIAGNOSTICS")
    print("=" * 60)

    run_case("ZERO ACTIONS", zero_actions)
    run_case("CONSTANT ACTIONS", constant_actions)
    run_case("ALTERNATING ACTIONS", alternating_actions)

    print("\nDIAGNOSTICS COMPLETE")


