
import numpy as np

from src.envs.dual_arm_transport_env import DualArmTransportEnv
from src.controllers.jacobian_action import calculate_cartesian_action


STEP_SIZE_M = 0.0005
TEST_STEPS = 100


def run_test(name, direction):
    env = DualArmTransportEnv(
        target_position=np.array([0.05, 0.0, 0.58]),
        success_distance=0.01,
    )

    env.reset()

    initial_position = env.get_object_state()[0].copy()

    left_missing = 0
    right_missing = 0
    max_action = 0.0

    terminated = False
    truncated = False
    info = {}
    steps = 0

    for step in range(TEST_STEPS):
        desired_displacement = (
            np.asarray(direction, dtype=np.float64)
            * STEP_SIZE_M
        )

        left_action = calculate_cartesian_action(
            env,
            env.left_ee_site_id,
            env.left_dof_ids,
            desired_displacement,
        )

        right_action = calculate_cartesian_action(
            env,
            env.right_ee_site_id,
            env.right_dof_ids,
            desired_displacement,
        )

        max_action = max(
            max_action,
            float(np.max(np.abs(left_action))),
            float(np.max(np.abs(right_action))),
        )

        _, _, terminated, truncated, info = env.step({
            "left": left_action,
            "right": right_action,
        })

        steps = step + 1

        left_contacts, right_contacts = env.get_grasp_contacts()

        if left_contacts == 0:
            left_missing += 1

        if right_contacts == 0:
            right_missing += 1

        if terminated or truncated:
            break

    final_position = env.get_object_state()[0].copy()
    displacement = final_position - initial_position

    left_contacts, right_contacts = env.get_grasp_contacts()

    print(f"\n{name}")
    print("-" * 55)
    print("Steps:", steps)
    print("Object displacement [mm]:", displacement * 1000)
    print("Max normalized joint action:", max_action)
    print("Missing left-contact steps:", left_missing)
    print("Missing right-contact steps:", right_missing)
    print("Final left contacts:", left_contacts)
    print("Final right contacts:", right_contacts)
    print("Success:", info.get("success"))
    print("Terminated:", terminated)
    print("Truncated:", truncated)


def main():
    print("=" * 55)
    print("CARTESIAN ACTION DIAGNOSTICS")
    print("=" * 55)

    tests = [
        ("X + Y", [0.7, 0.7, 0.0]),
        ("X - Y", [0.7, -0.7, 0.0]),
        ("X + Z", [0.7, 0.0, 0.7]),
        ("X - Z", [0.7, 0.0, -0.7]),
        ("X + Y + Z", [0.6, 0.6, 0.6]),
        ("X - Y - Z", [0.6, -0.6, -0.6]),
    ]

    for name, direction in tests:
        run_test(name, direction)

    print("\nDIAGNOSTICS COMPLETE")


if __name__ == "__main__":
    main()

