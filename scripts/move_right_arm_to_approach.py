from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = Path(
    "assets/models/dual_panda/transport_scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)


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
# Initial configurations
# ------------------------------------------------------------

left_initial_q = np.array([
    0.0,
    -0.4,
    0.0,
    -1.8,
    0.0,
    1.4,
    0.8,
])

right_initial_q = np.array([
    0.0,
    -0.4,
    0.0,
    -1.8,
    0.0,
    1.4,
    0.8,
])


# IK solution obtained from test_right_arm_ik.py
right_target_q = np.array([
    0.0,
    -0.30607,
    0.0,
    -1.80148,
    0.0,
    1.45606,
    0.8,
])


# ------------------------------------------------------------
# Get qpos addresses
# ------------------------------------------------------------

def get_qpos_addresses(joint_names):

    addresses = []

    for name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        addresses.append(
            model.jnt_qposadr[joint_id]
        )

    return addresses


left_qpos_ids = get_qpos_addresses(
    left_joint_names
)

right_qpos_ids = get_qpos_addresses(
    right_joint_names
)


# ------------------------------------------------------------
# Get actuator IDs by NAME
# ------------------------------------------------------------

def get_actuator_ids(prefix):

    ids = []

    for i in range(1, 8):

        name = f"{prefix}_actuator{i}"

        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

        if actuator_id == -1:
            raise RuntimeError(
                f"Actuator not found: {name}"
            )

        ids.append(actuator_id)

    return ids


left_actuator_ids = get_actuator_ids("left")
right_actuator_ids = get_actuator_ids("right")


# ------------------------------------------------------------
# Set initial robot states
# ------------------------------------------------------------

for qpos_id, value in zip(
    left_qpos_ids,
    left_initial_q,
):
    data.qpos[qpos_id] = value


for qpos_id, value in zip(
    right_qpos_ids,
    right_initial_q,
):
    data.qpos[qpos_id] = value


mujoco.mj_forward(model, data)


# ------------------------------------------------------------
# End-effector IDs
# ------------------------------------------------------------

left_ee_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "left_ee_site",
)

right_ee_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_ee_site",
)


print("Left actuator IDs:", left_actuator_ids)
print("Right actuator IDs:", right_actuator_ids)

print(
    "Initial left EE:",
    data.site_xpos[left_ee_id].copy(),
)

print(
    "Initial right EE:",
    data.site_xpos[right_ee_id].copy(),
)


# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

step = 0
transition_steps = 2000


with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    while viewer.is_running():

        # Gradually interpolate from initial configuration
        # to the IK solution.
        alpha = min(
            step / transition_steps,
            1.0,
        )

        desired_right_q = (
            (1.0 - alpha) * right_initial_q
            + alpha * right_target_q
        )

        # Left arm follows IK target
        for actuator_id, target in zip(
            left_actuator_ids,
            left_initial_q,
        ):
            data.ctrl[actuator_id] = target

        # Right arm stays at initial configuration
        for actuator_id, target in zip(
            right_actuator_ids,
            desired_right_q,
        ):
            data.ctrl[actuator_id] = target

        mujoco.mj_step(model, data)

        if step % 500 == 0:

            left_pos = (
                data.site_xpos[left_ee_id].copy()
            )

            right_pos = (
                data.site_xpos[right_ee_id].copy()
            )

            print(
                f"Step {step:5d} | "
                f"Left EE = {left_pos} | "
                f"Right EE = {right_pos}"
            )

        viewer.sync()

        time.sleep(model.opt.timestep)

        step += 1
