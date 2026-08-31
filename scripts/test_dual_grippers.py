from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = Path(
    "assets/models/dual_panda/transport_scene.xml"
)


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)


# ------------------------------------------------------------
# Nominal arm configuration
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
# Set both arms to nominal configuration
# ------------------------------------------------------------

for prefix in [
    "left",
    "right",
]:

    for i, value in enumerate(
        initial_q,
        start=1,
    ):

        joint_name = (
            f"{prefix}_joint{i}"
        )

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_id = model.jnt_qposadr[
            joint_id
        ]

        data.qpos[
            qpos_id
        ] = value


# ------------------------------------------------------------
# Finger joint IDs
# ------------------------------------------------------------

finger_joint_names = [
    "left_finger_joint1",
    "left_finger_joint2",
    "right_finger_joint1",
    "right_finger_joint2",
]


finger_qpos_ids = {}


for name in finger_joint_names:

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    qpos_id = model.jnt_qposadr[
        joint_id
    ]

    finger_qpos_ids[
        name
    ] = qpos_id


# ------------------------------------------------------------
# Gripper actuator IDs
# ------------------------------------------------------------

left_gripper_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "left_actuator8",
)

right_gripper_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "right_actuator8",
)


print(
    "Left gripper actuator:",
    left_gripper_id,
)

print(
    "Right gripper actuator:",
    right_gripper_id,
)


# ------------------------------------------------------------
# Arm actuator commands
# ------------------------------------------------------------

for prefix in [
    "left",
    "right",
]:

    for i, value in enumerate(
        initial_q,
        start=1,
    ):

        actuator_name = (
            f"{prefix}_actuator{i}"
        )

        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )

        data.ctrl[
            actuator_id
        ] = value


mujoco.mj_forward(
    model,
    data,
)


# ------------------------------------------------------------
# Diagnostic function
# ------------------------------------------------------------

def print_fingers(label):

    print("\n")
    print("=" * 60)
    print(label)
    print("=" * 60)

    for name in finger_joint_names:

        qpos_id = (
            finger_qpos_ids[
                name
            ]
        )

        print(
            f"{name}: "
            f"{data.qpos[qpos_id]:.6f}"
        )


# ------------------------------------------------------------
# Viewer
# ------------------------------------------------------------

with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    print_fingers(
        "INITIAL FINGER POSITIONS"
    )

    for _ in range(500):

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()


    print_fingers(
        "AFTER SETTLING"
    )


    # --------------------------------------------------------
    # Test ctrl = 0
    # --------------------------------------------------------

    print(
        "\nTesting both grippers "
        "with ctrl = 0..."
    )

    data.ctrl[
        left_gripper_id
    ] = 0.0

    data.ctrl[
        right_gripper_id
    ] = 0.0


    for _ in range(1000):

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()


    print_fingers(
        "AFTER CTRL = 0"
    )


    # --------------------------------------------------------
    # Test ctrl = 255
    # --------------------------------------------------------

    print(
        "\nTesting both grippers "
        "with ctrl = 255..."
    )

    data.ctrl[
        left_gripper_id
    ] = 255.0

    data.ctrl[
        right_gripper_id
    ] = 255.0


    for _ in range(1000):

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()


    print_fingers(
        "AFTER CTRL = 255"
    )


    # --------------------------------------------------------
    # Hold
    # --------------------------------------------------------

    print(
        "\nTest finished."
    )

    print(
        "Viewer will remain open."
    )


    while viewer.is_running():

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()

        time.sleep(
            model.opt.timestep
        )
