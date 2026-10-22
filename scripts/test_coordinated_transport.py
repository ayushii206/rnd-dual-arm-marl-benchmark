
from pathlib import Path
import sys
import argparse
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.dual_arm_transport_env import (
    DualArmTransportEnv,
    ACTION_SCALE_RAD,
)


TARGET_FORWARD_M = 0.01
MAX_STEPS = 1000
MAX_ACTION_MAGNITUDE = 0.25
DAMPING = 0.05

# Conservative velocity target in metres per RL step.
FORWARD_STEP_M = 0.0005


def quaternion_to_rotation(q):
    """MuJoCo quaternion [w, x, y, z] to rotation matrix."""
    matrix = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(q, dtype=np.float64))
    return matrix.reshape(3, 3)


def get_tracking_state(env):
    object_pos, object_quat, _, _ = env.get_object_state()
    rotation = quaternion_to_rotation(object_quat)

    left_ee = env.data.site_xpos[
        env.left_ee_site_id
    ].copy()

    right_ee = env.data.site_xpos[
        env.right_ee_site_id
    ].copy()

    return {
        "object": object_pos.copy(),
        "rotation": rotation,
        "left_relative": (
            rotation.T @ (left_ee - object_pos)
        ),
        "right_relative": (
            rotation.T @ (right_ee - object_pos)
        ),
    }


def calculate_arm_action(env, site_id, dof_ids):
    """
    Damped least-squares Jacobian inverse for a small
    positive-X end-effector displacement.

    Also penalizes unwanted angular movement.
    """
    nv = env.model.nv

    jac_pos = np.zeros((3, nv))
    jac_rot = np.zeros((3, nv))

    mujoco.mj_jacSite(
        env.model,
        env.data,
        jac_pos,
        jac_rot,
        site_id,
    )

    position_jacobian = jac_pos[:, dof_ids]
    rotation_jacobian = jac_rot[:, dof_ids]

    jacobian = np.vstack([
        position_jacobian,
        0.5 * rotation_jacobian,
    ])

    desired_motion = np.array([
        FORWARD_STEP_M,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    # Damped least-squares inverse:
    # dq = J.T @ inv(J J.T + lambda^2 I) @ dx
    regularized = (
        jacobian @ jacobian.T
        + (DAMPING ** 2) * np.eye(6)
    )

    joint_delta = jacobian.T @ np.linalg.solve(
        regularized,
        desired_motion,
    )

    action = joint_delta / ACTION_SCALE_RAD

    return np.clip(
        action,
        -MAX_ACTION_MAGNITUDE,
        MAX_ACTION_MAGNITUDE,
    )


def get_grasp_contacts(env):
    """
    Count MuJoCo contacts between each Panda's finger
    collision geometries and the transport object.

    Counts are instantaneous contact points, not forces.
    """

    left_finger_geoms = set(range(68, 84))
    right_finger_geoms = set(range(149, 165))
    object_geoms = set(range(165, 170))

    left_contacts = 0
    right_contacts = 0

    for i in range(env.data.ncon):
        contact = env.data.contact[i]

        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        pair = {geom1, geom2}

        if (
            pair & object_geoms
            and pair & left_finger_geoms
        ):
            left_contacts += 1

        if (
            pair & object_geoms
            and pair & right_finger_geoms
        ):
            right_contacts += 1

    return left_contacts, right_contacts


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=float,
        default=TARGET_FORWARD_M,
        help="Requested forward displacement in metres",
    )

    args = parser.parse_args()

    if not (0.0 < args.target <= 0.38):
        parser.error("Target must be between 0 and 0.38 m.")

    target_forward_m = args.target

    env = DualArmTransportEnv(
        target_position=np.array(
            [target_forward_m, 0.0, 0.58],
            dtype=np.float64,
        ),
        success_distance=0.01,
    )
    env.reset()

    initial = get_tracking_state(env)
    initial_object_x = initial["object"][0]

    max_left_drift = 0.0
    max_right_drift = 0.0

    left_contact_loss_steps = 0
    right_contact_loss_steps = 0

    initial_left_contacts, initial_right_contacts = (
        get_grasp_contacts(env)
    )

    print(
        "Initial finger-object contacts:",
        "left =", initial_left_contacts,
        "right =", initial_right_contacts,
    )

    print("=" * 70)
    print("COORDINATED TRANSPORT FEASIBILITY TEST")
    print("=" * 70)

    print("Initial object position:", initial["object"])
    print("Requested forward motion:", target_forward_m, "m")

    terminated = False
    truncated = False
    info = {}

    for step in range(1, MAX_STEPS + 1):
        left_action = calculate_arm_action(
            env,
            env.left_ee_site_id,
            env.left_dof_ids,
        )

        right_action = calculate_arm_action(
            env,
            env.right_ee_site_id,
            env.right_dof_ids,
        )

        (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        ) = env.step({
            "left": left_action,
            "right": right_action,
        })

        current = get_tracking_state(env)

        left_contacts, right_contacts = get_grasp_contacts(env)

        if left_contacts == 0:
            left_contact_loss_steps += 1

        if right_contacts == 0:
            right_contact_loss_steps += 1

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

        forward_displacement = (
            current["object"][0] - initial_object_x
        )

        if step % 25 == 0 or step == 1:
            print(
                f"Step {step:3d} | "
                f"object dx={forward_displacement:.5f} m | "
                f"left drift={left_drift:.5f} m | "
                f"right drift={right_drift:.5f} m"
                f" | left contacts={left_contacts}"
                f" | right contacts={right_contacts}"
            )

        if terminated or truncated:
            print("\nEnvironment episode ended.")
            break

        if forward_displacement >= target_forward_m:
            print("\nRequested forward displacement reached.")
            break

    final = get_tracking_state(env)

    object_displacement = (
        final["object"] - initial["object"]
    )

    rotation_difference = (
        initial["rotation"].T @ final["rotation"]
    )

    cos_angle = np.clip(
        (np.trace(rotation_difference) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    orientation_change_rad = float(
        np.arccos(cos_angle)
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("Object displacement [m]:", object_displacement)
    print("Forward displacement [mm]:",
          object_displacement[0] * 1000.0)
    print("Maximum left relative drift [mm]:",
          max_left_drift * 1000.0)
    print("Maximum right relative drift [mm]:",
          max_right_drift * 1000.0)
    print("Object orientation change [deg]:",
          np.degrees(orientation_change_rad))
    print("Dropped:", info.get("dropped"))
    print("Success:", info.get("success"))
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    print("\nCONTACT RETENTION RESULTS")
    print("-" * 70)

    print(
        "Initial left contacts:",
        initial_left_contacts,
    )

    print(
        "Initial right contacts:",
        initial_right_contacts,
    )

    print(
        "Final left contacts:",
        left_contacts,
    )

    print(
        "Final right contacts:",
        right_contacts,
    )

    print(
        "Steps without left finger-object contact:",
        left_contact_loss_steps,
    )

    print(
        "Steps without right finger-object contact:",
        right_contact_loss_steps,
    )

    print(
        "Final object height [m]:",
        final["object"][2],
    )

    print("=" * 70)
    print("COORDINATED TRANSPORT DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

