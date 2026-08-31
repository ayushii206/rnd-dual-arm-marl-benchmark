from pathlib import Path

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


# ------------------------------------------------------------
# Project setup
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.controllers.ik_controller import (
    solve_pose_ik,
)

from src.controllers.gravity_compensation import (
    get_joint_dof_ids,
    apply_bias_compensation,
)


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


# ------------------------------------------------------------
# Initial arm configuration
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
# Helper functions
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


def apply_arm_bias_compensation():
    """
    Apply feed-forward bias-force compensation
    to both Panda arms.

    This is called immediately before every
    MuJoCo physics step.
    """

    # Remove forces left from the previous step.
    data.qfrc_applied[:] = 0.0

    apply_bias_compensation(
        model,
        data,
        left_dof_ids,
    )

    apply_bias_compensation(
        model,
        data,
        right_dof_ids,
    )


def interpolate_arm_targets(
    start_left,
    end_left,
    start_right,
    end_right,
    steps,
    viewer,
):
    """
    Smoothly interpolate both arms from their
    start configurations to their target
    configurations.
    """

    for step in range(
        steps
    ):

        alpha = (
            step + 1
        ) / steps

        left_q = (
            (1.0 - alpha)
            * start_left
            + alpha
            * end_left
        )

        right_q = (
            (1.0 - alpha)
            * start_right
            + alpha
            * end_right
        )

        command_arm(
            left_actuator_ids,
            left_q,
        )

        command_arm(
            right_actuator_ids,
            right_q,
        )

        apply_arm_bias_compensation()

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()


def get_arm_qpos(
    joint_names,
):
    """
    Read the seven actual joint positions
    of one arm.
    """

    values = []

    for joint_name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_id = model.jnt_qposadr[
            joint_id
        ]

        values.append(
            data.qpos[
                qpos_id
            ]
        )

    return np.array(
        values
    )

def print_object_contacts():
    """
    Print all active contacts involving the transport object.

    For unnamed robot geoms, the owning body name is also printed,
    which lets us identify whether a finger, hand, or arm link is
    interfering with the object.
    """

    print("\n")
    print("=" * 70)
    print("ACTIVE CONTACTS INVOLVING TRANSPORT OBJECT")
    print("=" * 70)

    found_contact = False

    for contact_id in range(
        data.ncon
    ):

        contact = data.contact[
            contact_id
        ]

        geom1_id = contact.geom1
        geom2_id = contact.geom2

        body1_id = model.geom_bodyid[
            geom1_id
        ]

        body2_id = model.geom_bodyid[
            geom2_id
        ]

        geom1_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom1_id,
        )

        geom2_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom2_id,
        )

        body1_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body1_id,
        )

        body2_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body2_id,
        )

        object_involved = (
            body1_name == "transport_object"
            or body2_name == "transport_object"
        )

        if not object_involved:
            continue

        found_contact = True

        contact_force = np.zeros(
            6
        )

        mujoco.mj_contactForce(
            model,
            data,
            contact_id,
            contact_force,
        )

        print(
            f"\nContact {contact_id}"
        )

        print(
            f"  geom1 ID/name : "
            f"{geom1_id} / {geom1_name}"
        )

        print(
            f"  body1         : "
            f"{body1_name}"
        )

        print(
            f"  geom2 ID/name : "
            f"{geom2_id} / {geom2_name}"
        )

        print(
            f"  body2         : "
            f"{body2_name}"
        )

        print(
            f"  distance      : "
            f"{contact.dist}"
        )

        print(
            f"  normal force  : "
            f"{contact_force[0]}"
        )
        
        print(
            "  contact pos   :",
            contact.pos,
        )

        contact_frame = (
            contact.frame
            .reshape(
                3,
                3,
            )
        )

        print(
            "  contact normal:",
            contact_frame[0],
        )


    if not found_contact:

        print(
            "\nNo active contacts involving "
            "transport_object."
        )

    print(
        "\n" + "=" * 70
    )

def print_finger_joint_positions():
    """
    Print the current finger joint positions for
    both Panda grippers.
    """

    finger_joint_names = [
        "left_finger_joint1",
        "left_finger_joint2",
        "right_finger_joint1",
        "right_finger_joint2",
    ]

    print("\n")
    print("=" * 70)
    print("FINGER JOINT POSITIONS AFTER CLOSING")
    print("=" * 70)

    for joint_name in finger_joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_id = model.jnt_qposadr[
            joint_id
        ]

        print(
            f"{joint_name}: "
            f"{data.qpos[qpos_id]:.6f} m"
        )

    print(
        "=" * 70
    )

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


# Arm DOFs used for model-based bias compensation.
left_dof_ids = get_joint_dof_ids(
    model,
    left_actuator_ids,
)

right_dof_ids = get_joint_dof_ids(
    model,
    right_actuator_ids,
)


left_gripper_actuator_id = (
    mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "left_actuator8",
    )
)

right_gripper_actuator_id = (
    mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "right_actuator8",
    )
)


print(
    "Left arm actuator IDs:",
    left_actuator_ids,
)

print(
    "Right arm actuator IDs:",
    right_actuator_ids,
)

print(
    "Left gripper actuator ID:",
    left_gripper_actuator_id,
)

print(
    "Right gripper actuator ID:",
    right_gripper_actuator_id,
)


# ------------------------------------------------------------
# Put robots in initial configuration
# ------------------------------------------------------------

set_qpos(
    left_qpos_ids,
    initial_q,
)

set_qpos(
    right_qpos_ids,
    initial_q,
)

data.qvel[:] = 0.0

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
    data.site_xmat[
        left_site_id
    ]
    .reshape(
        3,
        3,
    )
    .copy()
)

right_target_rotation = (
    data.site_xmat[
        right_site_id
    ]
    .reshape(
        3,
        3,
    )
    .copy()
)


# ------------------------------------------------------------
# Cartesian grasp targets
# ------------------------------------------------------------

left_pregrasp_position = np.array([
    0.0,
    0.185,
    0.56,
])

left_grasp_position = np.array([
    0.0,
    0.145,
    0.56,
])


right_pregrasp_position = np.array([
    0.0,
    -0.185,
    0.56,
])

right_grasp_position = np.array([
    0.0,
    -0.145,
    0.56,
])


# ------------------------------------------------------------
# Solve IK
# ------------------------------------------------------------

left_pregrasp_q, _, _ = (
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


left_grasp_q, _, _ = (
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


right_pregrasp_q, _, _ = (
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


right_grasp_q, _, _ = (
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


print(
    "\nIK solved."
)

print(
    "Left pre-grasp q:",
    left_pregrasp_q,
)

print(
    "Left grasp q:",
    left_grasp_q,
)

print(
    "Right pre-grasp q:",
    right_pregrasp_q,
)

print(
    "Right grasp q:",
    right_grasp_q,
)


# ------------------------------------------------------------
# Reset after IK
# ------------------------------------------------------------

set_qpos(
    left_qpos_ids,
    initial_q,
)

set_qpos(
    right_qpos_ids,
    initial_q,
)

data.qvel[:] = 0.0

mujoco.mj_forward(
    model,
    data,
)


# ------------------------------------------------------------
# Initial arm commands
# ------------------------------------------------------------

command_arm(
    left_actuator_ids,
    initial_q,
)

command_arm(
    right_actuator_ids,
    initial_q,
)


# ------------------------------------------------------------
# Gripper commands
# ------------------------------------------------------------

OPEN_GRIPPER_CTRL = 255.0
CLOSE_GRIPPER_CTRL = 0.0


# ------------------------------------------------------------
# Run viewer
# ------------------------------------------------------------

with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    # --------------------------------------------------------
    # Open grippers
    # --------------------------------------------------------

    print(
        "\nOpening grippers..."
    )

    data.ctrl[
        left_gripper_actuator_id
    ] = OPEN_GRIPPER_CTRL

    data.ctrl[
        right_gripper_actuator_id
    ] = OPEN_GRIPPER_CTRL


    for _ in range(
        500
    ):

        apply_arm_bias_compensation()

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()


    # --------------------------------------------------------
    # Move to pre-grasp
    # --------------------------------------------------------

    print(
        "Moving to pre-grasp..."
    )

    interpolate_arm_targets(
        initial_q,
        left_pregrasp_q,
        initial_q,
        right_pregrasp_q,
        2000,
        viewer,
    )

    print(
        "Pre-grasp reached."
    )


    # Hold pre-grasp
    for _ in range(
        500
    ):

        apply_arm_bias_compensation()

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()


    # --------------------------------------------------------
    # Move from pre-grasp into grasp pose
    # --------------------------------------------------------

    print(
        "Moving into grasp..."
    )

    interpolate_arm_targets(
        left_pregrasp_q,
        left_grasp_q,
        right_pregrasp_q,
        right_grasp_q,
        1500,
        viewer,
    )

    print(
        "Grasp pose reached."
    )


    # --------------------------------------------------------
    # Settle at grasp pose before measuring
    # --------------------------------------------------------

    print(
        "Settling at grasp pose..."
    )

    for _ in range(
        1000
    ):

        command_arm(
            left_actuator_ids,
            left_grasp_q,
        )

        command_arm(
            right_actuator_ids,
            right_grasp_q,
        )

        apply_arm_bias_compensation()

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()


    print(
        "Grasp pose settled."
    )


    # --------------------------------------------------------
    # Update kinematics/contact information
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        data,
    )


    # --------------------------------------------------------
    # Diagnose contacts BEFORE gripper closing
    # --------------------------------------------------------

    print_object_contacts()

# --------------------------------------------------------
# Diagnose final grasp alignment
# --------------------------------------------------------


    left_grasp_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "left_grasp_site",
    )

    right_grasp_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "right_grasp_site",
    )


    left_ee_position = (
        data.site_xpos[
            left_site_id
        ].copy()
    )

    right_ee_position = (
        data.site_xpos[
            right_site_id
        ].copy()
    )


    left_handle_position = (
        data.site_xpos[
            left_grasp_site_id
        ].copy()
    )

    right_handle_position = (
        data.site_xpos[
            right_grasp_site_id
        ].copy()
    )


    print(
        "\n"
    )

    print(
        "=" * 60
    )

    print(
        "FINAL GRASP ALIGNMENT"
    )

    print(
        "=" * 60
    )


    print(
        "\nLEFT"
    )

    print(
        "EE position:",
        left_ee_position,
    )

    print(
        "Handle position:",
        left_handle_position,
    )

    print(
        "Handle - EE:",
        left_handle_position
        - left_ee_position,
    )

    print(
        "Distance:",
        np.linalg.norm(
            left_handle_position
            - left_ee_position
        ),
    )


    print(
        "\nRIGHT"
    )

    print(
        "EE position:",
        right_ee_position,
    )

    print(
        "Handle position:",
        right_handle_position,
    )

    print(
        "Handle - EE:",
        right_handle_position
        - right_ee_position,
    )

    print(
        "Distance:",
        np.linalg.norm(
            right_handle_position
            - right_ee_position
        ),
    )


    # --------------------------------------------------------
    # Diagnose actual joint tracking
    # --------------------------------------------------------

    left_actual_q = get_arm_qpos(
        left_joint_names
    )

    right_actual_q = get_arm_qpos(
        right_joint_names
    )


    left_q_error = (
        left_grasp_q
        - left_actual_q
    )

    right_q_error = (
        right_grasp_q
        - right_actual_q
    )


    print(
        "\n"
    )

    print(
        "=" * 60
    )

    print(
        "FINAL JOINT TRACKING"
    )

    print(
        "=" * 60
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


    # --------------------------------------------------------
    # Hold open grasp pose briefly
    # --------------------------------------------------------

    for _ in range(
        500
    ):

        # Continue explicitly commanding the grasp configuration.
        command_arm(
            left_actuator_ids,
            left_grasp_q,
        )

        command_arm(
            right_actuator_ids,
            right_grasp_q,
        )

        apply_arm_bias_compensation()

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()


    # --------------------------------------------------------
    # Close both grippers
    # --------------------------------------------------------

    print(
        "Closing grippers..."
    )


    data.ctrl[
        left_gripper_actuator_id
    ] = CLOSE_GRIPPER_CTRL

    data.ctrl[
        right_gripper_actuator_id
    ] = CLOSE_GRIPPER_CTRL


    for _ in range(
        1500
    ):

        # Keep the arms at their grasp targets
        # while the fingers close.
        command_arm(
            left_actuator_ids,
            left_grasp_q,
        )

        command_arm(
            right_actuator_ids,
            right_grasp_q,
        )

        apply_arm_bias_compensation()

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()


    print(
        "Grippers closed."
    )

    # --------------------------------------------------------
    # Diagnose contacts AFTER gripper closing
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        data,
    )

    print(
        "\n\nPOST-CLOSING CONTACT DIAGNOSTIC"
    )

    print_object_contacts()
    
    print_finger_joint_positions()

    # --------------------------------------------------------
    # Hold final grasp
    # --------------------------------------------------------

    print(
        "\nHolding final pose..."
    )


    while viewer.is_running():

        command_arm(
            left_actuator_ids,
            left_grasp_q,
        )

        command_arm(
            right_actuator_ids,
            right_grasp_q,
        )

        apply_arm_bias_compensation()

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()

        time.sleep(
            model.opt.timestep
        )
