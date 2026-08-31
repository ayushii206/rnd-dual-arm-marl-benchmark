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
# Initial configuration
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


left_joint_names = [
    f"left_joint{i}"
    for i in range(1, 8)
]

right_joint_names = [
    f"right_joint{i}"
    for i in range(1, 8)
]


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
# Set nominal configuration
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
# Save nominal EE orientations
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


# ------------------------------------------------------------
# Pre-grasp and grasp targets
# ------------------------------------------------------------

left_pregrasp_position = np.array([
    0.0,
    0.185,
    0.56,
])

left_grasp_position = np.array([
    0.0,
    0.135,
    0.56,
])


right_pregrasp_position = np.array([
    0.0,
    -0.185,
    0.56,
])

right_grasp_position = np.array([
    0.0,
    -0.135,
    0.56,
])


# ------------------------------------------------------------
# Solve LEFT pre-grasp
# ------------------------------------------------------------

left_pregrasp_q, left_pregrasp_final, _ = (
    solve_pose_ik(
        model=model,
        data=data,
        joint_names=left_joint_names,
        site_name="left_ee_site",
        target_position=left_pregrasp_position,
        target_rotation=left_target_rotation,
        initial_q=initial_q,
    )
)


# ------------------------------------------------------------
# Solve LEFT grasp
# ------------------------------------------------------------

left_grasp_q, left_grasp_final, _ = (
    solve_pose_ik(
        model=model,
        data=data,
        joint_names=left_joint_names,
        site_name="left_ee_site",
        target_position=left_grasp_position,
        target_rotation=left_target_rotation,
        initial_q=left_pregrasp_q,
    )
)


# ------------------------------------------------------------
# Solve RIGHT pre-grasp
# ------------------------------------------------------------

right_pregrasp_q, right_pregrasp_final, _ = (
    solve_pose_ik(
        model=model,
        data=data,
        joint_names=right_joint_names,
        site_name="right_ee_site",
        target_position=right_pregrasp_position,
        target_rotation=right_target_rotation,
        initial_q=initial_q,
    )
)


# ------------------------------------------------------------
# Solve RIGHT grasp
# ------------------------------------------------------------

right_grasp_q, right_grasp_final, _ = (
    solve_pose_ik(
        model=model,
        data=data,
        joint_names=right_joint_names,
        site_name="right_ee_site",
        target_position=right_grasp_position,
        target_rotation=right_target_rotation,
        initial_q=right_pregrasp_q,
    )
)


# ------------------------------------------------------------
# Errors
# ------------------------------------------------------------

left_pregrasp_error = np.linalg.norm(
    left_pregrasp_position
    - left_pregrasp_final
)

left_grasp_error = np.linalg.norm(
    left_grasp_position
    - left_grasp_final
)

right_pregrasp_error = np.linalg.norm(
    right_pregrasp_position
    - right_pregrasp_final
)

right_grasp_error = np.linalg.norm(
    right_grasp_position
    - right_grasp_final
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("LEFT ARM")
print("=" * 60)

print("\nPre-grasp target:")
print(left_pregrasp_position)

print("\nPre-grasp final:")
print(left_pregrasp_final)

print("\nPre-grasp error:")
print(left_pregrasp_error)

print("\nPre-grasp joint solution:")
print(left_pregrasp_q)

print("\nGrasp target:")
print(left_grasp_position)

print("\nGrasp final:")
print(left_grasp_final)

print("\nGrasp error:")
print(left_grasp_error)

print("\nGrasp joint solution:")
print(left_grasp_q)


print("\n")
print("=" * 60)
print("RIGHT ARM")
print("=" * 60)

print("\nPre-grasp target:")
print(right_pregrasp_position)

print("\nPre-grasp final:")
print(right_pregrasp_final)

print("\nPre-grasp error:")
print(right_pregrasp_error)

print("\nPre-grasp joint solution:")
print(right_pregrasp_q)

print("\nGrasp target:")
print(right_grasp_position)

print("\nGrasp final:")
print(right_grasp_final)

print("\nGrasp error:")
print(right_grasp_error)

print("\nGrasp joint solution:")
print(right_grasp_q)
