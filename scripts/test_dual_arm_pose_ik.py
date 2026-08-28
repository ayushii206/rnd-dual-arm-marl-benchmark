from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))


from src.controllers.ik_controller import (
    solve_pose_ik,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "assets/models/dual_panda/transport_scene.xml"
)


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)


# ------------------------------------------------------------
# Initial joint configuration
# ------------------------------------------------------------

initial_q = np.array([
    0.0,
    -0.4,
    0.0,
    -1.8,
    0.0,
    1.4,
    0.8,
])


# ------------------------------------------------------------
# Joint names
# ------------------------------------------------------------

left_joint_names = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
]


right_joint_names = [
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7",
]


# ------------------------------------------------------------
# Helper: set one arm configuration
# ------------------------------------------------------------

def set_arm_configuration(
    joint_names,
    q,
):

    for name, value in zip(
        joint_names,
        q,
    ):

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        qpos_id = model.jnt_qposadr[
            joint_id
        ]

        data.qpos[qpos_id] = value


# ------------------------------------------------------------
# Put both robots into initial configuration
# ------------------------------------------------------------

set_arm_configuration(
    left_joint_names,
    initial_q,
)

set_arm_configuration(
    right_joint_names,
    initial_q,
)

mujoco.mj_forward(
    model,
    data,
)


# ------------------------------------------------------------
# End-effector site IDs
# ------------------------------------------------------------

left_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "left_ee_site",
)

right_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_ee_site",
)


# ------------------------------------------------------------
# Save INITIAL orientations
#
# These become our desired orientations.
# We want the arms to move in XYZ without changing
# their current wrist orientations.
# ------------------------------------------------------------

left_target_rotation = (
    data.site_xmat[left_site_id]
    .reshape(3, 3)
    .copy()
)

right_target_rotation = (
    data.site_xmat[right_site_id]
    .reshape(3, 3)
    .copy()
)


print("Desired left orientation:")
print(left_target_rotation)

print("\nDesired right orientation:")
print(right_target_rotation)


# ------------------------------------------------------------
# Cartesian approach targets
# ------------------------------------------------------------

left_target_position = np.array([
    0.0,
    0.21,
    0.56,
])

right_target_position = np.array([
    0.0,
    -0.21,
    0.56,
])


# ------------------------------------------------------------
# Solve LEFT pose IK
# ------------------------------------------------------------

left_solution, left_final_position, left_final_rotation = (
    solve_pose_ik(
        model=model,
        data=data,
        joint_names=left_joint_names,
        site_name="left_ee_site",
        target_position=left_target_position,
        target_rotation=left_target_rotation,
        initial_q=initial_q,
    )
)


# ------------------------------------------------------------
# Solve RIGHT pose IK
# ------------------------------------------------------------

right_solution, right_final_position, right_final_rotation = (
    solve_pose_ik(
        model=model,
        data=data,
        joint_names=right_joint_names,
        site_name="right_ee_site",
        target_position=right_target_position,
        target_rotation=right_target_rotation,
        initial_q=initial_q,
    )
)


# ------------------------------------------------------------
# Compute final position errors
# ------------------------------------------------------------

left_position_error = np.linalg.norm(
    left_target_position
    - left_final_position
)

right_position_error = np.linalg.norm(
    right_target_position
    - right_final_position
)


# ------------------------------------------------------------
# Compute rotation-matrix differences
# ------------------------------------------------------------

left_rotation_difference = (
    left_target_rotation
    - left_final_rotation
)

right_rotation_difference = (
    right_target_rotation
    - right_final_rotation
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("LEFT ARM")
print("=" * 60)

print("\nJoint solution:")
print(left_solution)

print("\nTarget position:")
print(left_target_position)

print("\nFinal position:")
print(left_final_position)

print("\nPosition error magnitude:")
print(left_position_error)

print("\nTarget rotation:")
print(left_target_rotation)

print("\nFinal rotation:")
print(left_final_rotation)

print("\nRotation matrix difference:")
print(left_rotation_difference)


print("\n")
print("=" * 60)
print("RIGHT ARM")
print("=" * 60)

print("\nJoint solution:")
print(right_solution)

print("\nTarget position:")
print(right_target_position)

print("\nFinal position:")
print(right_final_position)

print("\nPosition error magnitude:")
print(right_position_error)

print("\nTarget rotation:")
print(right_target_rotation)

print("\nFinal rotation:")
print(right_final_rotation)

print("\nRotation matrix difference:")
print(right_rotation_difference)
