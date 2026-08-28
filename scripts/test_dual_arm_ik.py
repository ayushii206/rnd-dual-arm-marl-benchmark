from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))


from src.controllers.ik_controller import (
    solve_position_ik,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "assets/models/dual_panda/transport_scene.xml"
)

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)


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


left_target = np.array([
    0.0,
    0.21,
    0.56,
])


right_target = np.array([
    0.0,
    -0.21,
    0.56,
])


left_solution, left_position, left_error = (
    solve_position_ik(
        model=model,
        data=data,
        joint_names=left_joint_names,
        site_name="left_ee_site",
        target_position=left_target,
        initial_q=initial_q,
    )
)


right_solution, right_position, right_error = (
    solve_position_ik(
        model=model,
        data=data,
        joint_names=right_joint_names,
        site_name="right_ee_site",
        target_position=right_target,
        initial_q=initial_q,
    )
)


print("LEFT ARM")
print("Solution:")
print(left_solution)

print("Final position:")
print(left_position)

print("Error:")
print(left_error)


print("\nRIGHT ARM")
print("Solution:")
print(right_solution)

print("Final position:")
print(right_position)

print("Error:")
print(right_error)
