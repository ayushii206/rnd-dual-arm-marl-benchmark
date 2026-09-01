from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "assets/models/dual_panda/transport_scene.xml"
)


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    model
)


print("\n")
print("=" * 70)
print("GRIPPER ACTUATOR PARAMETERS")
print("=" * 70)


for actuator_name in [
    "left_actuator8",
    "right_actuator8",
]:

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )

    print(
        f"\n{actuator_name}"
    )

    print(
        "Actuator ID:",
        actuator_id,
    )

    print(
        "ctrlrange:",
        model.actuator_ctrlrange[
            actuator_id
        ],
    )

    print(
        "forcerange:",
        model.actuator_forcerange[
            actuator_id
        ],
    )

    print(
        "gainprm:",
        model.actuator_gainprm[
            actuator_id
        ],
    )

    print(
        "biasprm:",
        model.actuator_biasprm[
            actuator_id
        ],
    )

    print(
        "gear:",
        model.actuator_gear[
            actuator_id
        ],
    )


print("\n")
print("=" * 70)
print("FINGER JOINT PARAMETERS")
print("=" * 70)


for joint_name in [
    "left_finger_joint1",
    "left_finger_joint2",
    "right_finger_joint1",
    "right_finger_joint2",
]:

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    qpos_id = model.jnt_qposadr[
        joint_id
    ]

    dof_id = model.jnt_dofadr[
        joint_id
    ]

    print(
        f"\n{joint_name}"
    )

    print(
        "Joint ID:",
        joint_id,
    )

    print(
        "qpos ID:",
        qpos_id,
    )

    print(
        "DOF ID:",
        dof_id,
    )

    print(
        "range:",
        model.jnt_range[
            joint_id
        ],
    )


print(
    "\n" + "=" * 70
)
