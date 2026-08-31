from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.controllers.ik_controller import solve_pose_ik


MODEL_PATH = (
    PROJECT_ROOT
    / "assets/models/dual_panda/transport_scene.xml"
)

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)


# ------------------------------------------------------------
# Arm configuration
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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def get_qpos_ids(
    joint_names,
):

    ids = []

    for name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        ids.append(
            model.jnt_qposadr[
                joint_id
            ]
        )

    return ids


def get_actuator_ids(
    prefix,
):

    ids = []

    for i in range(
        1,
        8,
    ):

        actuator_name = (
            f"{prefix}_actuator{i}"
        )

        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )

        ids.append(
            actuator_id
        )

    return ids


def set_qpos(
    qpos_ids,
    q,
):

    for qpos_id, value in zip(
        qpos_ids,
        q,
    ):

        data.qpos[
            qpos_id
        ] = value


def command_arm(
    actuator_ids,
    q,
):

    for actuator_id, value in zip(
        actuator_ids,
        q,
    ):

        data.ctrl[
            actuator_id
        ] = value


def get_arm_qpos(
    qpos_ids,
):

    return np.array([
        data.qpos[
            qpos_id
        ]
        for qpos_id
        in qpos_ids
    ])


# ------------------------------------------------------------
# IDs
# ------------------------------------------------------------

left_qpos_ids = get_qpos_ids(
    left_joint_names
)

right_qpos_ids = get_qpos_ids(
    right_joint_names
)


left_actuator_ids = get_actuator_ids(
    "left"
)

right_actuator_ids = get_actuator_ids(
    "right"
)


left_gripper_actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "left_actuator8",
)

right_gripper_actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "right_actuator8",
)


# ------------------------------------------------------------
# Initial state
# ------------------------------------------------------------

set_qpos(
    left_qpos_ids,
    initial_q,
)

set_qpos(
    right_qpos_ids,
    initial_q,
)


# Open grippers directly
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

    data.qpos[
        qpos_id
    ] = 0.04


mujoco.mj_forward(
    model,
    data,
)


# ------------------------------------------------------------
# Nominal EE orientations
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


left_rotation = (
    data.site_xmat[
        left_site_id
    ]
    .reshape(3, 3)
    .copy()
)

right_rotation = (
    data.site_xmat[
        right_site_id
    ]
    .reshape(3, 3)
    .copy()
)


# ------------------------------------------------------------
# Grasp targets
# ------------------------------------------------------------

left_grasp_target = np.array([
    0.0,
    0.135,
    0.56,
])

right_grasp_target = np.array([
    0.0,
    -0.135,
    0.56,
])


# ------------------------------------------------------------
# Solve IK
# ------------------------------------------------------------

left_result = solve_pose_ik(
    model=model,
    data=data,
    joint_names=left_joint_names,
    site_name="left_ee_site",
    target_position=left_grasp_target,
    target_rotation=left_rotation,
    initial_q=initial_q,
)

right_result = solve_pose_ik(
    model=model,
    data=data,
    joint_names=right_joint_names,
    site_name="right_ee_site",
    target_position=right_grasp_target,
    target_rotation=right_rotation,
    initial_q=initial_q,
)


left_grasp_q = np.asarray(
    left_result[0]
    if isinstance(
        left_result,
        tuple,
    )
    else left_result,
    dtype=float,
).reshape(-1)

right_grasp_q = np.asarray(
    right_result[0]
    if isinstance(
        right_result,
        tuple,
    )
    else right_result,
    dtype=float,
).reshape(-1)


print(
    "\nLeft grasp q:",
    left_grasp_q,
)

print(
    "Right grasp q:",
    right_grasp_q,
)


# ------------------------------------------------------------
# Reset simulation after IK
# ------------------------------------------------------------

mujoco.mj_resetData(
    model,
    data,
)


set_qpos(
    left_qpos_ids,
    initial_q,
)

set_qpos(
    right_qpos_ids,
    initial_q,
)


mujoco.mj_forward(
    model,
    data,
)


# ------------------------------------------------------------
# Freeze transport object from step zero
# ------------------------------------------------------------

object_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "object_freejoint",
)

object_qpos_adr = model.jnt_qposadr[
    object_joint_id
]

object_dof_adr = model.jnt_dofadr[
    object_joint_id
]


initial_object_qpos = data.qpos[
    object_qpos_adr:
    object_qpos_adr + 7
].copy()


# ------------------------------------------------------------
# Disable transport object collisions
# ------------------------------------------------------------

object_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "transport_object",
)


for geom_id in range(
    model.ngeom
):

    if (
        model.geom_bodyid[
            geom_id
        ]
        == object_body_id
    ):

        model.geom_contype[
            geom_id
        ] = 0

        model.geom_conaffinity[
            geom_id
        ] = 0


# ------------------------------------------------------------
# Tracking test
# ------------------------------------------------------------

OPEN_GRIPPER_CTRL = 255.0


for step in range(
    6000
):

    command_arm(
        left_actuator_ids,
        left_grasp_q,
    )

    command_arm(
        right_actuator_ids,
        right_grasp_q,
    )


    data.ctrl[
        left_gripper_actuator_id
    ] = OPEN_GRIPPER_CTRL

    data.ctrl[
        right_gripper_actuator_id
    ] = OPEN_GRIPPER_CTRL


    # --------------------------------------------------------
    # Model-based bias-force compensation
    # --------------------------------------------------------

    data.qfrc_applied[:] = 0.0

    for actuator_id in left_actuator_ids:

        joint_id = model.actuator_trnid[
            actuator_id,
            0,
        ]

        dof_id = model.jnt_dofadr[
            joint_id
        ]

        data.qfrc_applied[
            dof_id
        ] = data.qfrc_bias[
            dof_id
        ]

    for actuator_id in right_actuator_ids:

        joint_id = model.actuator_trnid[
            actuator_id,
            0,
        ]

        dof_id = model.jnt_dofadr[
            joint_id
        ]

        data.qfrc_applied[
            dof_id
        ] = data.qfrc_bias[
            dof_id
        ]


    mujoco.mj_step(
        model,
        data,
    )


    # Freeze object after every physics step.
    data.qpos[
        object_qpos_adr:
        object_qpos_adr + 7
    ] = initial_object_qpos

    data.qvel[
        object_dof_adr:
        object_dof_adr + 6
    ] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    # Freeze object after every physics step.
    data.qpos[
        object_qpos_adr:
        object_qpos_adr + 7
    ] = initial_object_qpos

    data.qvel[
        object_dof_adr:
        object_dof_adr + 6
    ] = 0.0


    mujoco.mj_forward(
        model,
        data,
    )


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

left_actual_q = get_arm_qpos(
    left_qpos_ids
)

right_actual_q = get_arm_qpos(
    right_qpos_ids
)


left_q_error = (
    left_grasp_q
    - left_actual_q
)

right_q_error = (
    right_grasp_q
    - right_actual_q
)


left_ee = data.site_xpos[
    left_site_id
].copy()

right_ee = data.site_xpos[
    right_site_id
].copy()


left_handle_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "left_grasp_site",
)

right_handle_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_grasp_site",
)


left_handle = data.site_xpos[
    left_handle_id
].copy()

right_handle = data.site_xpos[
    right_handle_id
].copy()


print(
    "\n"
    + "=" * 70
)

print(
    "TRACKING TEST WITHOUT OBJECT CONTACT"
)

print(
    "=" * 70
)


print(
    "\nLEFT"
)

print(
    "Commanded q:",
    left_grasp_q,
)

print(
    "Actual q:",
    left_actual_q,
)

print(
    "q error:",
    left_q_error,
)

print(
    "q error norm:",
    np.linalg.norm(
        left_q_error
    ),
)

print(
    "EE position:",
    left_ee,
)

print(
    "Handle position:",
    left_handle,
)

print(
    "EE-handle distance:",
    np.linalg.norm(
        left_handle
        - left_ee
    ),
)


print(
    "\nRIGHT"
)

print(
    "Commanded q:",
    right_grasp_q,
)

print(
    "Actual q:",
    right_actual_q,
)

print(
    "q error:",
    right_q_error,
)

print(
    "q error norm:",
    np.linalg.norm(
        right_q_error
    ),
)

print(
    "EE position:",
    right_ee,
)

print(
    "Handle position:",
    right_handle,
)

print(
    "EE-handle distance:",
    np.linalg.norm(
        right_handle
        - right_ee
    ),
)
