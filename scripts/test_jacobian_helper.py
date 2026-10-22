
import numpy as np

from src.envs.dual_arm_transport_env import DualArmTransportEnv
from src.controllers.jacobian_action import calculate_cartesian_action


def main():
    env = DualArmTransportEnv(
        target_position=np.array([0.05, 0.0, 0.58]),
        success_distance=0.01,
    )

    env.reset()

    initial_position = env.get_object_state()[0].copy()

    final_info = {}
    terminated = False
    truncated = False
    steps = 0

    for step in range(500):
        left_action = calculate_cartesian_action(
            env,
            env.left_ee_site_id,
            env.left_dof_ids,
            np.array([0.0005, 0.0, 0.0]),
        )

        right_action = calculate_cartesian_action(
            env,
            env.right_ee_site_id,
            env.right_dof_ids,
            np.array([0.0005, 0.0, 0.0]),
        )

        _, _, terminated, truncated, final_info = env.step({
            "left": left_action,
            "right": right_action,
        })

        steps = step + 1

        if terminated or truncated:
            break

    final_position = env.get_object_state()[0].copy()

    left_contacts, right_contacts = env.get_grasp_contacts()

    print("=" * 60)
    print("JACOBIAN HELPER TEST")
    print("=" * 60)
    print("Steps:", steps)
    print("Displacement:", final_position - initial_position)
    print("Final target distance:",
          final_info["distance_to_target"])
    print("Left contacts:", left_contacts)
    print("Right contacts:", right_contacts)
    print("Success:", final_info["success"])
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    assert final_info["success"], (
        "Jacobian helper did not reach the target."
    )

    assert left_contacts > 0 and right_contacts > 0, (
        "Cooperative grasp was lost."
    )

    print("JACOBIAN HELPER: PASS")


if __name__ == "__main__":
    main()

