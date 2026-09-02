from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


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


# ============================================================
# Configuration
# ============================================================

INITIAL_ARM_Q = np.array([
    0.0,
    -0.4,
    0.0,
    -1.8,
    0.0,
    1.4,
    0.8,
])


LEFT_PREGRASP_POSITION = np.array([
    0.0,
    0.185,
    0.56,
])

RIGHT_PREGRASP_POSITION = np.array([
    0.0,
    -0.185,
    0.56,
])


LEFT_GRASP_POSITION = np.array([
    0.0,
    0.145,
    0.56,
])

RIGHT_GRASP_POSITION = np.array([
    0.0,
    -0.145,
    0.56,
])


# First lift:
# raise both end effectors exactly 2 cm.

LEFT_LIFT_POSITION = np.array([
    0.0,
    0.145,
    0.58,
])

RIGHT_LIFT_POSITION = np.array([
    0.0,
    -0.145,
    0.58,
])

# ============================================================
# Cooperative goal-transport configuration
# ============================================================

# Desired object-center translation toward the goal.
GOAL_TRANSPORT_DISTANCE_X = 0.45

# Extra vertical clearance used only for the long
# tilted transport to the goal.
#
# The validated grasp/lift still ends at Z = 0.58 m.
GOAL_TRANSPORT_Z = 0.592

LEFT_CLEARANCE_POSITION = np.array([
    0.0,
    0.145,
    GOAL_TRANSPORT_Z,
])

RIGHT_CLEARANCE_POSITION = np.array([
    0.0,
    -0.145,
    GOAL_TRANSPORT_Z,
])

# A common world-frame rotation about Y is required for the
# bilateral grasp to remain kinematically reachable at X=0.45.
#
# The refined IK scan found:
#   -8.6 deg -> FAIL
#   -8.7 deg -> PASS, but almost exactly at tolerance
#   -9.0 deg -> comfortable PASS
#
# Therefore use -9.0 deg for the dynamic validation.
GOAL_COMMON_Y_ROTATION_DEG = -9.0


# Number of Cartesian rigid-body waypoints used to construct
# the cooperative transport path.
#
# These are IK waypoints, not MuJoCo physics steps.
GOAL_TRAJECTORY_WAYPOINTS = 46


# Number of physics steps used between each pair of
# consecutive IK waypoints.
PHYSICS_STEPS_PER_WAYPOINT_SEGMENT = 100


# Total dynamic transport duration.
GOAL_TRANSPORT_STEPS = (
    (GOAL_TRAJECTORY_WAYPOINTS - 1)
    * PHYSICS_STEPS_PER_WAYPOINT_SEGMENT
)


# IK validation tolerances.
GOAL_IK_POSITION_TOLERANCE_M = 1.0e-3
GOAL_IK_ORIENTATION_TOLERANCE_DEG = 1.0

OPEN_GRIPPER_CTRL = 255.0
CLOSE_GRIPPER_CTRL = 0.0


# ============================================================
# Names
# ============================================================

left_joint_names = [
    f"left_joint{i}"
    for i in range(1, 8)
]

right_joint_names = [
    f"right_joint{i}"
    for i in range(1, 8)
]


# ============================================================
# Helper functions
# ============================================================

def get_qpos_ids(
    joint_names,
):
    qpos_ids = []

    for joint_name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_id = model.jnt_qposadr[
            joint_id
        ]

        qpos_ids.append(
            qpos_id
        )

    return qpos_ids


def get_actuator_ids(
    prefix,
):
    actuator_ids = []

    for actuator_id in range(
        model.nu
    ):

        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        if (
            actuator_name is not None
            and actuator_name.startswith(
                prefix
            )
        ):

            actuator_ids.append(
                actuator_id
            )

    return actuator_ids


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
    for actuator_id, target in zip(
        actuator_ids,
        q,
    ):

        data.ctrl[
            actuator_id
        ] = target


def get_arm_qpos(
    qpos_ids,
):
    return np.array([
        data.qpos[
            qpos_id
        ]
        for qpos_id in qpos_ids
    ])


def apply_arm_bias_compensation():
    """
    Apply MuJoCo generalized bias-force
    feed-forward compensation to both arms.
    """

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


def step_with_arm_targets(
    left_q,
    right_q,
):
    """
    Execute one physics step while maintaining
    both arm joint targets.
    """

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

def rotation_y_world(
    angle_deg,
):
    """
    Return a 3x3 active rotation matrix for a rotation
    about the MuJoCo world Y axis.
    """

    angle_rad = np.radians(
        angle_deg
    )

    c = np.cos(
        angle_rad
    )

    s = np.sin(
        angle_rad
    )

    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def orientation_error_deg(
    target_rotation,
    achieved_rotation,
):
    """
    Magnitude of orientation error between two rotation
    matrices, expressed in degrees.
    """

    rotation_difference = (
        target_rotation.T
        @ achieved_rotation
    )

    cos_angle = np.clip(
        (
            np.trace(
                rotation_difference
            )
            - 1.0
        )
        / 2.0,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                cos_angle
            )
        )
    )


def smoothstep(
    alpha,
):
    """
    Cubic smoothstep.

    Unlike a linear trajectory, this starts and ends with
    zero trajectory velocity.
    """

    return (
        alpha
        * alpha
        * (
            3.0
            - 2.0 * alpha
        )
    )

def interpolate_arm_targets(
    viewer,
    left_start_q,
    left_target_q,
    right_start_q,
    right_target_q,
    steps,
):
    """
    Smoothly interpolate both arm targets.
    """

    for step in range(
        steps
    ):

        alpha = (
            step + 1
        ) / steps

        left_q = (
            (1.0 - alpha)
            * left_start_q
            + alpha
            * left_target_q
        )

        right_q = (
            (1.0 - alpha)
            * right_start_q
            + alpha
            * right_target_q
        )

        step_with_arm_targets(
            left_q,
            right_q,
        )

        viewer.sync()


def print_finger_joint_positions():
    finger_joint_names = [
        "left_finger_joint1",
        "left_finger_joint2",
        "right_finger_joint1",
        "right_finger_joint2",
    ]

    print("\n")
    print("=" * 70)
    print("FINGER JOINT POSITIONS")
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


def print_object_contacts(
    title,
):
    """
    Print every active contact involving the transport object.

    For each contact:
    - print the MuJoCo contact-frame force
    - transform the force into world coordinates
    - determine the force acting ON the transport object
    - accumulate vertical support force

    This allows us to distinguish normal gripping force
    from the tangential friction force that can actually
    support the object against gravity.
    """

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    found_contact = False

    total_object_force_world = np.zeros(3)
    total_finger_force_world = np.zeros(3)
    total_table_force_world = np.zeros(3)

    total_finger_vertical_force = 0.0
    total_table_vertical_force = 0.0

    finger_contact_count = 0
    table_contact_count = 0

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

        if (
            body1_name
            != "transport_object"
            and body2_name
            != "transport_object"
        ):
            continue

        found_contact = True

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

        # ----------------------------------------------------
        # Contact force in MuJoCo contact coordinates
        # ----------------------------------------------------

        contact_wrench = np.zeros(
            6
        )

        mujoco.mj_contactForce(
            model,
            data,
            contact_id,
            contact_wrench,
        )

        contact_force_local = (
            contact_wrench[
                0:3
            ].copy()
        )

        # ----------------------------------------------------
        # Contact frame
        #
        # MuJoCo stores the contact frame as:
        #
        # row 0 = contact normal
        # row 1 = first tangent direction
        # row 2 = second tangent direction
        #
        # Therefore a force expressed in contact coordinates
        # is transformed to world coordinates using frame.T.
        # ----------------------------------------------------

        contact_frame = (
            contact.frame
            .reshape(
                3,
                3,
            )
        )

        # mj_contactForce follows MuJoCo's contact convention:
        # the contact force points from geom1 toward geom2.
        #
        # Therefore the transformed force is the force acting
        # on geom2. The equal-and-opposite force acts on geom1.

        force_world_on_geom2 = (
            contact_frame.T
            @ contact_force_local
        )

        force_world_on_geom1 = (
            -force_world_on_geom2
        )

        # ----------------------------------------------------
        # Determine force acting specifically on the object
        # ----------------------------------------------------

        if (
            body1_name
            == "transport_object"
        ):

            object_force_world = (
                force_world_on_geom1
            )

            other_body_name = (
                body2_name
            )

        else:

            object_force_world = (
                force_world_on_geom2
            )

            other_body_name = (
                body1_name
            )

        total_object_force_world += (
            object_force_world
        )

        # ----------------------------------------------------
        # Classify support source
        # ----------------------------------------------------

        is_finger_contact = (
            other_body_name
            is not None
            and "finger"
            in other_body_name.lower()
        )

        is_table_contact = (
            other_body_name
            == "world"
            and (
                geom1_name
                == "work_table"
                or geom2_name
                == "work_table"
            )
        )

        if is_finger_contact:

            finger_contact_count += 1

            total_finger_force_world += (
                object_force_world
            )

            total_finger_vertical_force += (
                object_force_world[2]
            )

        if is_table_contact:

            table_contact_count += 1

            total_table_force_world += (
                object_force_world
            )

            total_table_vertical_force += (
                object_force_world[2]
            )

        # ----------------------------------------------------
        # Individual contact diagnostic
        # ----------------------------------------------------

        print(
            f"\nContact {contact_id}"
        )

        print(
            "  geom1:",
            geom1_id,
            "/",
            geom1_name,
        )

        print(
            "  body1:",
            body1_name,
        )

        print(
            "  geom2:",
            geom2_id,
            "/",
            geom2_name,
        )

        print(
            "  body2:",
            body2_name,
        )

        print(
            "  distance:",
            contact.dist,
        )

        print(
            "  contact position:",
            contact.pos,
        )

        print(
            "  contact normal:",
            contact_frame[0],
        )
        
        print(
            "  effective contact friction:",
            contact.friction,
        )
        
        print(
            "  contact dimension:",
            contact.dim,
        )

        print(
            "  force in contact frame:",
            contact_force_local,
        )

        print(
            "  normal component:",
            contact_force_local[0],
        )

        print(
            "  tangent component 1:",
            contact_force_local[1],
        )

        print(
            "  tangent component 2:",
            contact_force_local[2],
        )

        print(
            "  force on object [world XYZ]:",
            object_force_world,
        )

        print(
            "  vertical force on object [world Z]:",
            object_force_world[2],
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    if not found_contact:

        print(
            "\nNo contacts involving transport_object."
        )

    else:

        print("\n")
        print("-" * 70)
        print("CONTACT FORCE SUMMARY")
        print("-" * 70)

        print(
            "Finger contact count:",
            finger_contact_count,
        )

        print(
            "Table contact count:",
            table_contact_count,
        )

        print(
            "\nTotal finger force on object [world XYZ]:",
            total_finger_force_world,
        )

        print(
            "Total finger vertical force on object [N]:",
            total_finger_vertical_force,
        )

        print(
            "\nTotal table force on object [world XYZ]:",
            total_table_force_world,
        )

        print(
            "Total table vertical force on object [N]:",
            total_table_vertical_force,
        )

        print(
            "\nTotal contact force on object [world XYZ]:",
            total_object_force_world,
        )

        print(
            "Total contact vertical force on object [N]:",
            total_object_force_world[2],
        )

        print(
            "\nExpected object weight for 2.0 kg [N]:",
            2.0 * 9.81,
        )

    print(
        "\n" + "=" * 70
    )
    
def get_object_support_forces():
    """
    Return the instantaneous vertical contact forces acting
    on the transport object.

    Returns
    -------
    finger_fz : float
        Total world-Z force from finger contacts.

    table_fz : float
        Total world-Z force from table contacts.

    finger_normal_sum : float
        Sum of finger-handle contact normal-force magnitudes.

    finger_contact_count : int
        Number of active finger-object contacts.
    """

    finger_fz = 0.0
    table_fz = 0.0

    finger_normal_sum = 0.0
    finger_contact_count = 0

    for contact_id in range(data.ncon):

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

        if (
            body1_name != "transport_object"
            and body2_name != "transport_object"
        ):
            continue

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

        contact_wrench = np.zeros(6)

        mujoco.mj_contactForce(
            model,
            data,
            contact_id,
            contact_wrench,
        )

        contact_force_local = (
            contact_wrench[:3]
        )

        contact_frame = (
            contact.frame.reshape(
                3,
                3,
            )
        )

        # Force acting on geom2.
        force_world_on_geom2 = (
            contact_frame.T
            @ contact_force_local
        )

        # Equal and opposite force on geom1.
        force_world_on_geom1 = (
            -force_world_on_geom2
        )

        if body1_name == "transport_object":

            object_force_world = (
                force_world_on_geom1
            )

            other_body_name = (
                body2_name
            )

        else:

            object_force_world = (
                force_world_on_geom2
            )

            other_body_name = (
                body1_name
            )

        is_finger_contact = (
            other_body_name is not None
            and "finger"
            in other_body_name.lower()
        )

        is_table_contact = (
            other_body_name == "world"
            and (
                geom1_name == "work_table"
                or geom2_name == "work_table"
            )
        )

        if is_finger_contact:

            finger_contact_count += 1

            finger_fz += (
                object_force_world[2]
            )

            finger_normal_sum += abs(
                contact_force_local[0]
            )

        if is_table_contact:

            table_fz += (
                object_force_world[2]
            )

    return (
        finger_fz,
        table_fz,
        finger_normal_sum,
        finger_contact_count,
    )
    
def object_state():
    """
    Return object position, rotation matrix,
    linear velocity and angular velocity.
    """

    object_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "transport_object",
    )

    position = data.xpos[
        object_body_id
    ].copy()

    rotation = (
        data.xmat[
            object_body_id
        ]
        .reshape(
            3,
            3,
        )
        .copy()
    )

    object_joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "object_freejoint",
    )

    dof_address = model.jnt_dofadr[
        object_joint_id
    ]

    linear_velocity = data.qvel[
        dof_address:
        dof_address + 3
    ].copy()

    angular_velocity = data.qvel[
        dof_address + 3:
        dof_address + 6
    ].copy()

    return (
        position,
        rotation,
        linear_velocity,
        angular_velocity,
    )


def rotation_matrix_to_rpy(
    rotation,
):
    """
    Convert a rotation matrix to roll/pitch/yaw.
    Used only for diagnostics.
    """

    pitch = np.arcsin(
        np.clip(
            -rotation[2, 0],
            -1.0,
            1.0,
        )
    )

    roll = np.arctan2(
        rotation[2, 1],
        rotation[2, 2],
    )

    yaw = np.arctan2(
        rotation[1, 0],
        rotation[0, 0],
    )

    return np.array([
        roll,
        pitch,
        yaw,
    ])


def print_object_state(
    title,
):
    (
        position,
        rotation,
        linear_velocity,
        angular_velocity,
    ) = object_state()

    rpy = rotation_matrix_to_rpy(
        rotation
    )

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        "Object position:",
        position,
    )

    print(
        "Object RPY [rad]:",
        rpy,
    )

    print(
        "Object RPY [deg]:",
        np.degrees(
            rpy
        ),
    )

    print(
        "Linear velocity:",
        linear_velocity,
    )

    print(
        "Angular velocity:",
        angular_velocity,
    )

    print(
        "=" * 70
    )

def print_gripper_actuator_state(
    title,
):
    """
    Print the actual MuJoCo actuator state for
    both grippers at the current simulation state.
    """

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    for actuator_id in [
        left_gripper_actuator_id,
        right_gripper_actuator_id,
    ]:

        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        print(
            f"\n{actuator_name}"
        )

        print(
            "ctrl:",
            data.ctrl[
                actuator_id
            ],
        )

        print(
            "actuator length:",
            data.actuator_length[
                actuator_id
            ],
        )

        print(
            "actuator velocity:",
            data.actuator_velocity[
                actuator_id
            ],
        )

        print(
            "actuator force:",
            data.actuator_force[
                actuator_id
            ],
        )

    print(
        "\n" + "=" * 70
    )
    
def print_gripper_transmission_state(
    title,
):
    """
    Inspect how each gripper actuator force is transmitted
    through its fixed tendon to the two finger joint DOFs.
    """

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    grippers = [
        {
            "label": "LEFT GRIPPER",
            "actuator_id": left_gripper_actuator_id,
            "tendon_name": "left_split",
            "finger_joint_names": [
                "left_finger_joint1",
                "left_finger_joint2",
            ],
        },
        {
            "label": "RIGHT GRIPPER",
            "actuator_id": right_gripper_actuator_id,
            "tendon_name": "right_split",
            "finger_joint_names": [
                "right_finger_joint1",
                "right_finger_joint2",
            ],
        },
    ]

    tendon_jacobian_values = np.asarray(
        data.ten_J
    )

    print(
        "data.ten_J shape:",
        tendon_jacobian_values.shape,
    )

    print(
        "model.ten_J_rownnz:",
        np.asarray(
            model.ten_J_rownnz
        ),
    )

    print(
        "model.ten_J_rowadr:",
        np.asarray(
            model.ten_J_rowadr
        ),
    )

    print(
        "model.ten_J_colind:",
        np.asarray(
            model.ten_J_colind
        ),
    )

    for gripper in grippers:

        print("\n")
        print("-" * 70)
        print(
            gripper["label"]
        )
        print("-" * 70)

        actuator_id = gripper[
            "actuator_id"
        ]

        tendon_name = gripper[
            "tendon_name"
        ]

        tendon_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            tendon_name,
        )

        print(
            "Actuator ID:",
            actuator_id,
        )

        print(
            "Actuator force:",
            data.actuator_force[
                actuator_id
            ],
        )

        print(
            "Actuator length:",
            data.actuator_length[
                actuator_id
            ],
        )

        print(
            "Actuator velocity:",
            data.actuator_velocity[
                actuator_id
            ],
        )

        print(
            "\nTendon name:",
            tendon_name,
        )

        print(
            "Tendon ID:",
            tendon_id,
        )

        print(
            "Tendon length:",
            data.ten_length[
                tendon_id
            ],
        )

        print(
            "Tendon velocity:",
            data.ten_velocity[
                tendon_id
            ],
        )

# ----------------------------------------------------
# Reconstruct this tendon Jacobian row from MuJoCo's
# sparse tendon Jacobian representation.
#
# In MuJoCo 3.8.1:
#
# data.ten_J
#     stores only the non-zero Jacobian values.
#
# model.ten_J_rowadr
#     gives the starting index for each tendon row.
#
# model.ten_J_rownnz
#     gives the number of non-zero entries in the row.
#
# model.ten_J_colind
#     gives the corresponding DOF column indices.
# ----------------------------------------------------

        row_start = int(
            model.ten_J_rowadr[
                tendon_id
            ]
        )

        row_nnz = int(
            model.ten_J_rownnz[
                tendon_id
            ]
        )

        row_end = (
            row_start
            + row_nnz
        )

        row_columns = np.asarray(
            model.ten_J_colind[
                row_start:row_end
            ],
            dtype=int,
        )

        row_values = np.asarray(
            data.ten_J[
                row_start:row_end
            ]
        )

        tendon_jacobian_row = np.zeros(
            model.nv
        )

        tendon_jacobian_row[
            row_columns
        ] = row_values

        print(
            "\nSparse tendon Jacobian:"
        )

        print(
            "  row start:",
            row_start,
        )

        print(
            "  non-zero count:",
            row_nnz,
        )

        print(
            "  DOF columns:",
            row_columns,
        )

        print(
            "  values:",
            row_values,
        )
        
        print(
            "\nFinger transmission:"
        )

        actuator_force = (
            data.actuator_force[
                actuator_id
            ]
        )

        for joint_name in gripper[
            "finger_joint_names"
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

            tendon_coefficient = (
                tendon_jacobian_row[
                    dof_id
                ]
            )

            predicted_generalized_force = (
                actuator_force
                * tendon_coefficient
            )

            actual_generalized_force = (
                data.qfrc_actuator[
                    dof_id
                ]
            )

            print(
                f"\n{joint_name}"
            )

            print(
                "  Joint ID:",
                joint_id,
            )

            print(
                "  qpos ID:",
                qpos_id,
            )

            print(
                "  DOF ID:",
                dof_id,
            )

            print(
                "  joint position:",
                data.qpos[
                    qpos_id
                ],
            )

            print(
                "  tendon Jacobian coefficient:",
                tendon_coefficient,
            )

            print(
                "  actuator_force * coefficient:",
                predicted_generalized_force,
            )

            print(
                "  data.qfrc_actuator:",
                actual_generalized_force,
            )

        print(
            "\nFull non-zero tendon Jacobian entries:"
        )

        nonzero_dofs = np.where(
            np.abs(
                tendon_jacobian_row
            )
            > 1e-10
        )[0]

        for dof_id in nonzero_dofs:

            print(
                f"  DOF {dof_id}: "
                f"{tendon_jacobian_row[dof_id]}"
            )

    print(
        "\n" + "=" * 70
    )
    
# ============================================================
# IDs
# ============================================================

left_qpos_ids = get_qpos_ids(
    left_joint_names
)

right_qpos_ids = get_qpos_ids(
    right_joint_names
)


left_actuator_ids = get_actuator_ids(
    "left_actuator"
)[:7]

right_actuator_ids = get_actuator_ids(
    "right_actuator"
)[:7]


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


left_dof_ids = get_joint_dof_ids(
    model,
    left_actuator_ids,
)

right_dof_ids = get_joint_dof_ids(
    model,
    right_actuator_ids,
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


# ============================================================
# Initial arm pose
# ============================================================

set_qpos(
    left_qpos_ids,
    INITIAL_ARM_Q,
)

set_qpos(
    right_qpos_ids,
    INITIAL_ARM_Q,
)

mujoco.mj_forward(
    model,
    data,
)


# ============================================================
# Preserve nominal EE orientations
# ============================================================

left_ee_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "left_ee_site",
)

right_ee_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_ee_site",
)


left_target_rotation = (
    data.site_xmat[
        left_ee_site_id
    ]
    .reshape(
        3,
        3,
    )
    .copy()
)

right_target_rotation = (
    data.site_xmat[
        right_ee_site_id
    ]
    .reshape(
        3,
        3,
    )
    .copy()
)


# ============================================================
# Solve IK
# ============================================================

left_pregrasp_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=left_joint_names,
    site_name="left_ee_site",
    target_position=LEFT_PREGRASP_POSITION,
    target_rotation=left_target_rotation,
    initial_q=INITIAL_ARM_Q,
)


right_pregrasp_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=right_joint_names,
    site_name="right_ee_site",
    target_position=RIGHT_PREGRASP_POSITION,
    target_rotation=right_target_rotation,
    initial_q=INITIAL_ARM_Q,
)


left_grasp_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=left_joint_names,
    site_name="left_ee_site",
    target_position=LEFT_GRASP_POSITION,
    target_rotation=left_target_rotation,
    initial_q=left_pregrasp_q,
)


right_grasp_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=right_joint_names,
    site_name="right_ee_site",
    target_position=RIGHT_GRASP_POSITION,
    target_rotation=right_target_rotation,
    initial_q=right_pregrasp_q,
)


left_lift_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=left_joint_names,
    site_name="left_ee_site",
    target_position=LEFT_LIFT_POSITION,
    target_rotation=left_target_rotation,
    initial_q=left_grasp_q,
)


right_lift_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=right_joint_names,
    site_name="right_ee_site",
    target_position=RIGHT_LIFT_POSITION,
    target_rotation=right_target_rotation,
    initial_q=right_grasp_q,
)

# ============================================================
# Solve extra clearance-lift IK
# ============================================================

left_clearance_q, left_clearance_achieved_position, _ = (
    solve_pose_ik(
        model=model,
        data=data,
        joint_names=left_joint_names,
        site_name="left_ee_site",
        target_position=LEFT_CLEARANCE_POSITION,
        target_rotation=left_target_rotation,
        initial_q=left_lift_q,
    )
)

right_clearance_q, right_clearance_achieved_position, _ = (
    solve_pose_ik(
        model=model,
        data=data,
        joint_names=right_joint_names,
        site_name="right_ee_site",
        target_position=RIGHT_CLEARANCE_POSITION,
        target_rotation=right_target_rotation,
        initial_q=right_lift_q,
    )
)


left_clearance_position_error = float(
    np.linalg.norm(
        left_clearance_achieved_position
        - LEFT_CLEARANCE_POSITION
    )
)

right_clearance_position_error = float(
    np.linalg.norm(
        right_clearance_achieved_position
        - RIGHT_CLEARANCE_POSITION
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "CLEARANCE-LIFT IK"
)

print(
    "=" * 70
)

print(
    "Left clearance target:",
    LEFT_CLEARANCE_POSITION,
)

print(
    "Left clearance achieved:",
    left_clearance_achieved_position,
)

print(
    "Left clearance error [mm]:",
    left_clearance_position_error * 1000.0,
)

print(
    "Right clearance target:",
    RIGHT_CLEARANCE_POSITION,
)

print(
    "Right clearance achieved:",
    right_clearance_achieved_position,
)

print(
    "Right clearance error [mm]:",
    right_clearance_position_error * 1000.0,
)

print(
    "=" * 70
)


if (
    left_clearance_position_error
    > GOAL_IK_POSITION_TOLERANCE_M
    or right_clearance_position_error
    > GOAL_IK_POSITION_TOLERANCE_M
):
    raise RuntimeError(
        "Clearance-lift IK failed position tolerance."
    )

# ============================================================
# Build cooperative rigid-body goal trajectory
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "BUILDING COOPERATIVE GOAL TRAJECTORY"
)

print(
    "=" * 70
)


# ------------------------------------------------------------
# The goal trajectory begins from the extra-clearance pose:
#
#   left EE  = [0, +0.145, 0.605]
#   right EE = [0, -0.145, 0.605]
#
# Their midpoint is the rigid-body reference point used
# for the cooperative goal trajectory.
# ------------------------------------------------------------

GRASP_REFERENCE_POSITION = (
    0.5
    * (
        LEFT_CLEARANCE_POSITION
        + RIGHT_CLEARANCE_POSITION
    )
)


LEFT_GRASP_OFFSET = (
    LEFT_CLEARANCE_POSITION
    - GRASP_REFERENCE_POSITION
)


RIGHT_GRASP_OFFSET = (
    RIGHT_CLEARANCE_POSITION
    - GRASP_REFERENCE_POSITION
)

print(
    "Grasp reference position:",
    GRASP_REFERENCE_POSITION,
)

print(
    "Left rigid grasp offset:",
    LEFT_GRASP_OFFSET,
)

print(
    "Right rigid grasp offset:",
    RIGHT_GRASP_OFFSET,
)


# ------------------------------------------------------------
# Storage for validated IK trajectory
# ------------------------------------------------------------

left_goal_trajectory_q = [
    left_clearance_q.copy()
]

right_goal_trajectory_q = [
    right_clearance_q.copy()
]


goal_trajectory_records = []


previous_left_q = (
    left_clearance_q.copy()
)

previous_right_q = (
    right_clearance_q.copy()
)


goal_trajectory_valid = True


# ------------------------------------------------------------
# Sequential continuation IK
#
# Waypoint 0 is the extra-clearance configuration, so start
# solving from waypoint 1.
# ------------------------------------------------------------

for waypoint_index in range(
    1,
    GOAL_TRAJECTORY_WAYPOINTS,
):

    raw_alpha = (
        waypoint_index
        / (
            GOAL_TRAJECTORY_WAYPOINTS
            - 1
        )
    )


    trajectory_alpha = smoothstep(
        raw_alpha
    )


    current_x = (
        GOAL_TRANSPORT_DISTANCE_X
        * trajectory_alpha
    )


    current_angle_deg = (
        GOAL_COMMON_Y_ROTATION_DEG
        * trajectory_alpha
    )


    common_rotation = rotation_y_world(
        current_angle_deg
    )


    # --------------------------------------------------------
    # Translate the rigid grasp reference point.
    #
    # Y and Z of the reference point remain unchanged.
    # --------------------------------------------------------

    current_reference_position = (
        GRASP_REFERENCE_POSITION
        + np.array([
            current_x,
            0.0,
            0.0,
        ])
    )


    # --------------------------------------------------------
    # Rotate BOTH grasp offsets using exactly the same
    # world-frame rotation.
    #
    # This guarantees rigid bilateral position compatibility.
    # --------------------------------------------------------

    left_target_position = (
        current_reference_position
        + common_rotation
        @ LEFT_GRASP_OFFSET
    )


    right_target_position = (
        current_reference_position
        + common_rotation
        @ RIGHT_GRASP_OFFSET
    )


    # --------------------------------------------------------
    # Rotate BOTH nominal EE orientations using exactly the
    # same world-frame rotation.
    # --------------------------------------------------------

    left_goal_target_rotation = (
        common_rotation
        @ left_target_rotation
    )

    right_goal_target_rotation = (
        common_rotation
        @ right_target_rotation
    )


    # --------------------------------------------------------
    # Sequential IK.
    #
    # Each solution is seeded by the preceding solution.
    # --------------------------------------------------------

    (
        left_waypoint_q,
        left_achieved_position,
        left_achieved_rotation,
    ) = solve_pose_ik(
        model=model,
        data=data,
        joint_names=left_joint_names,
        site_name="left_ee_site",
        target_position=left_target_position,
        target_rotation=left_goal_target_rotation,
        initial_q=previous_left_q,
    )


    (
        right_waypoint_q,
        right_achieved_position,
        right_achieved_rotation,
    ) = solve_pose_ik(
        model=model,
        data=data,
        joint_names=right_joint_names,
        site_name="right_ee_site",
        target_position=right_target_position,
        target_rotation=right_goal_target_rotation,
        initial_q=previous_right_q,
    )


    # --------------------------------------------------------
    # Validate waypoint
    # --------------------------------------------------------

    left_position_error = float(
        np.linalg.norm(
            left_achieved_position
            - left_target_position
        )
    )


    right_position_error = float(
        np.linalg.norm(
            right_achieved_position
            - right_target_position
        )
    )


    left_orientation_error = (
        orientation_error_deg(
            left_goal_target_rotation,
            left_achieved_rotation,
        )
    )


    right_orientation_error = (
        orientation_error_deg(
            right_goal_target_rotation,
            right_achieved_rotation,
        )
    )


    waypoint_pass = all([
        left_position_error
        <= GOAL_IK_POSITION_TOLERANCE_M,

        right_position_error
        <= GOAL_IK_POSITION_TOLERANCE_M,

        left_orientation_error
        <= GOAL_IK_ORIENTATION_TOLERANCE_DEG,

        right_orientation_error
        <= GOAL_IK_ORIENTATION_TOLERANCE_DEG,
    ])


    goal_trajectory_records.append({
        "index": waypoint_index,
        "raw_alpha": raw_alpha,
        "trajectory_alpha": trajectory_alpha,
        "x": current_x,
        "angle_deg": current_angle_deg,
        "left_position_error": (
            left_position_error
        ),
        "right_position_error": (
            right_position_error
        ),
        "left_orientation_error": (
            left_orientation_error
        ),
        "right_orientation_error": (
            right_orientation_error
        ),
        "pass": waypoint_pass,
    })


    print(
        f"Waypoint {waypoint_index:2d} | "
        f"x={current_x:.4f} m | "
        f"angle={current_angle_deg:+.3f} deg | "
        f"Lpos={left_position_error * 1000:.3f} mm | "
        f"Rpos={right_position_error * 1000:.3f} mm | "
        f"Lori={left_orientation_error:.4f} deg | "
        f"Rori={right_orientation_error:.4f} deg | "
        f"PASS={waypoint_pass}"
    )


    if not waypoint_pass:

        goal_trajectory_valid = False

        print(
            "\nGOAL TRAJECTORY IK FAILED."
        )

        print(
            "Failed waypoint:",
            waypoint_index,
        )

        break


    left_goal_trajectory_q.append(
        left_waypoint_q.copy()
    )

    right_goal_trajectory_q.append(
        right_waypoint_q.copy()
    )


    previous_left_q = (
        left_waypoint_q.copy()
    )

    previous_right_q = (
        right_waypoint_q.copy()
    )


# ------------------------------------------------------------
# Stop before physics if the planned rigid trajectory itself
# is not kinematically valid.
# ------------------------------------------------------------

if not goal_trajectory_valid:

    raise RuntimeError(
        "Cooperative goal trajectory failed IK validation. "
        "Dynamic transport was NOT started."
    )


if (
    len(left_goal_trajectory_q)
    != GOAL_TRAJECTORY_WAYPOINTS
    or len(right_goal_trajectory_q)
    != GOAL_TRAJECTORY_WAYPOINTS
):

    raise RuntimeError(
        "Goal trajectory contains an unexpected number "
        "of IK waypoints."
    )


left_transport_q = (
    left_goal_trajectory_q[-1].copy()
)

right_transport_q = (
    right_goal_trajectory_q[-1].copy()
)


print(
    "\n"
    + "=" * 70
)

print(
    "GOAL TRAJECTORY IK RESULT: PASS"
)

print(
    "Number of validated configurations:",
    len(left_goal_trajectory_q),
)

print(
    "Final common world-Y rotation [deg]:",
    GOAL_COMMON_Y_ROTATION_DEG,
)

print(
    "Final left q:",
    left_transport_q,
)

print(
    "Final right q:",
    right_transport_q,
)

print(
    "=" * 70
)


 #============================================================
# Reset simulation after IK
# ============================================================

mujoco.mj_resetData(
    model,
    data,
)

set_qpos(
    left_qpos_ids,
    INITIAL_ARM_Q,
)

set_qpos(
    right_qpos_ids,
    INITIAL_ARM_Q,
)

mujoco.mj_forward(
    model,
    data,
)


command_arm(
    left_actuator_ids,
    INITIAL_ARM_Q,
)

command_arm(
    right_actuator_ids,
    INITIAL_ARM_Q,
)

data.ctrl[
    left_gripper_actuator_id
] = OPEN_GRIPPER_CTRL

data.ctrl[
    right_gripper_actuator_id
] = OPEN_GRIPPER_CTRL


# ============================================================
# Viewer / experiment
# ============================================================

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

    for _ in range(
        500
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = OPEN_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = OPEN_GRIPPER_CTRL

        step_with_arm_targets(
            INITIAL_ARM_Q,
            INITIAL_ARM_Q,
        )

        viewer.sync()


    # --------------------------------------------------------
    # Move to pre-grasp
    # --------------------------------------------------------

    print(
        "Moving to pre-grasp..."
    )

    interpolate_arm_targets(
        viewer,
        INITIAL_ARM_Q,
        left_pregrasp_q,
        INITIAL_ARM_Q,
        right_pregrasp_q,
        2000,
    )

    print(
        "Pre-grasp reached."
    )


    # --------------------------------------------------------
    # Settle at pre-grasp
    # --------------------------------------------------------

    for _ in range(
        500
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = OPEN_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = OPEN_GRIPPER_CTRL

        step_with_arm_targets(
            left_pregrasp_q,
            right_pregrasp_q,
        )

        viewer.sync()


    # --------------------------------------------------------
    # Move into grasp pose
    # --------------------------------------------------------

    print(
        "Moving into grasp..."
    )

    interpolate_arm_targets(
        viewer,
        left_pregrasp_q,
        left_grasp_q,
        right_pregrasp_q,
        right_grasp_q,
        1500,
    )

    print(
        "Grasp pose reached."
    )


    # --------------------------------------------------------
    # Settle before closing
    # --------------------------------------------------------

    print(
        "Settling at grasp pose..."
    )

    for _ in range(
        1000
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = OPEN_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = OPEN_GRIPPER_CTRL

        step_with_arm_targets(
            left_grasp_q,
            right_grasp_q,
        )

        viewer.sync()


    print(
        "Grasp pose settled."
    )


    # --------------------------------------------------------
    # State before closing
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        data,
    )

    print_object_state(
        "OBJECT STATE BEFORE GRIPPER CLOSING"
    )

    print_object_contacts(
        "CONTACTS BEFORE GRIPPER CLOSING"
    )


    # --------------------------------------------------------
    # Close both grippers
    # --------------------------------------------------------

    print(
        "\nClosing grippers..."
    )

    for _ in range(
        2000
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        step_with_arm_targets(
            left_grasp_q,
            right_grasp_q,
        )

        viewer.sync()


    print(
        "Grippers closed."
    )


    # --------------------------------------------------------
    # Settle closed grasp
    # --------------------------------------------------------

    print(
        "Settling closed grasp..."
    )

    for _ in range(
        1000
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        step_with_arm_targets(
            left_grasp_q,
            right_grasp_q,
        )

        viewer.sync()


    mujoco.mj_forward(
        model,
        data,
    )


    print_finger_joint_positions()

    print_gripper_actuator_state(
        "GRIPPER ACTUATOR STATE BEFORE LIFT"
    )
    
    print_gripper_transmission_state(
    "GRIPPER TRANSMISSION STATE BEFORE LIFT"
)

    print_object_state(
        "OBJECT STATE BEFORE LIFT"
    )

    print_object_contacts(
        "CONTACTS BEFORE LIFT"
    )


    object_position_before_lift, _, _, _ = (
        object_state()
    )


    # --------------------------------------------------------
    # Cooperative 2 cm lift
    # --------------------------------------------------------

    print(
        "\nStarting cooperative 2 cm lift..."
    )

# --------------------------------------------------------
# Instrumented cooperative lift
#
# Record how vertical support changes WHILE the fingers
# move upward, not only after the lift has finished.
# --------------------------------------------------------

    LIFT_STEPS = 2500
    LOG_INTERVAL = 100

    object_center_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "object_center",
    )
    
    object_freejoint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "object_freejoint",
    )

    object_freejoint_dof_adr = model.jnt_dofadr[
        object_freejoint_id
    ]

    max_finger_fz = -np.inf
    max_finger_fz_step = -1

    max_finger_normal_sum = -np.inf
    max_finger_normal_step = -1

    min_table_fz = np.inf
    min_table_fz_step = -1

    max_object_z = -np.inf
    max_object_z_step = -1

    initial_object_z = float(
        data.site_xpos[
            object_center_site_id,
            2,
        ]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DYNAMIC LIFT FORCE DIAGNOSTIC"
    )

    print(
        "=" * 70
    )

    print(
        "Initial object Z:",
        initial_object_z,
    )

    print(
        "\n"
        "step | progress | object_z | "
        "delta_z | object_vz | finger_Fz | table_Fz | "
        "normal_sum | finger_contacts"
    )

    print(
        "-" * 110
    )

    for step_index in range(
        LIFT_STEPS
    ):

        # Progress goes from approximately 0 to exactly 1.
        alpha = (
            (step_index + 1)
            / LIFT_STEPS
        )

        left_target_q = (
            left_grasp_q
            + alpha
            * (
                left_lift_q
                - left_grasp_q
            )
        )

        right_target_q = (
            right_grasp_q
            + alpha
            * (
                right_lift_q
                - right_grasp_q
            )
        )

        # Keep both grippers fully closed throughout
        # the entire upward trajectory.
        data.ctrl[
            left_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        # This uses the same arm stepping / bias compensation
        # already used everywhere else in this experiment.
        step_with_arm_targets(
            left_target_q,
            right_target_q,
        )

        viewer.sync()

        # ----------------------------------------------------
        # Measure contact support AFTER this simulation step.
        # ----------------------------------------------------

        (
            finger_fz,
            table_fz,
            finger_normal_sum,
            finger_contact_count,
        ) = get_object_support_forces()

        object_z = float(
            data.site_xpos[
                object_center_site_id,
                2,
            ]
        )

        object_delta_z = (
            object_z
            - initial_object_z
        )
        
        object_vz = float(
            data.qvel[
                object_freejoint_dof_adr + 2
            ]
        )

        # ----------------------------------------------------
        # Track extrema over the complete 2500-step motion.
        # ----------------------------------------------------

        if (
            finger_fz
            > max_finger_fz
        ):

            max_finger_fz = (
                finger_fz
            )

            max_finger_fz_step = (
                step_index + 1
            )

        if (
            finger_normal_sum
            > max_finger_normal_sum
        ):

            max_finger_normal_sum = (
                finger_normal_sum
            )

            max_finger_normal_step = (
                step_index + 1
            )

        if (
            table_fz
            < min_table_fz
        ):

            min_table_fz = (
                table_fz
            )

            min_table_fz_step = (
                step_index + 1
            )

        if (
            object_z
            > max_object_z
        ):

            max_object_z = (
                object_z
            )

            max_object_z_step = (
                step_index + 1
            )

        # ----------------------------------------------------
        # Print every 100 steps, plus first and final step.
        # ----------------------------------------------------

        if (
            step_index == 0
            or (
                step_index + 1
            ) % LOG_INTERVAL == 0
            or step_index
            == LIFT_STEPS - 1
        ):

            print(
                f"{step_index + 1:4d} | "
                f"{alpha:8.4f} | "
                f"{object_z:9.6f} | "
                f"{object_delta_z:+9.6f} | "
                f"{object_vz:+10.6f} | "
                f"{finger_fz:+9.4f} | "
                f"{table_fz:+9.4f} | "
                f"{finger_normal_sum:10.4f} | "
                f"{finger_contact_count:3d}"
            )

    print(
        "\n"
        + "-" * 70
    )

    print(
        "DYNAMIC LIFT SUMMARY"
    )

    print(
        "-" * 70
    )

    print(
        "Initial object Z:",
        initial_object_z,
    )

    print(
        "Final object Z:",
        float(
            data.site_xpos[
                object_center_site_id,
                2,
            ]
        ),
    )
    
    print(
        "Object vertical velocity at end of lift [m/s]:",
        float(
            data.qvel[
                object_freejoint_dof_adr + 2
            ]
        ),
    )

    print(
        "Maximum object Z:",
        max_object_z,
    )

    print(
        "Maximum object vertical displacement:",
        max_object_z
        - initial_object_z,
    )

    print(
        "Maximum object Z occurred at step:",
        max_object_z_step,
    )

    print(
        "\nMaximum finger vertical force [N]:",
        max_finger_fz,
    )

    print(
        "Maximum finger vertical force occurred at step:",
        max_finger_fz_step,
    )

    print(
        "\nMaximum summed finger normal force [N]:",
        max_finger_normal_sum,
    )

    print(
        "Maximum summed finger normal force occurred at step:",
        max_finger_normal_step,
    )

    print(
        "\nMinimum table vertical support [N]:",
        min_table_fz,
    )

    print(
        "Minimum table support occurred at step:",
        min_table_fz_step,
    )

    print(
        "\nObject weight [N]:",
        2.0 * 9.81,
    )

    print(
        "=" * 70
    )

    print(
        "Lift target reached."
    )
    
    # ========================================================
    # Extra clearance lift
    # ========================================================

    print(
        "\nStarting extra 25 mm clearance lift..."
    )

    CLEARANCE_LIFT_STEPS = 1500

    first_clearance_lift_table_contact_step = None


    for clearance_step in range(
        CLEARANCE_LIFT_STEPS
    ):

        clearance_alpha = (
            clearance_step + 1
        ) / CLEARANCE_LIFT_STEPS


        left_clearance_target_q = (
            (1.0 - clearance_alpha)
            * left_lift_q
            + clearance_alpha
            * left_clearance_q
        )


        right_clearance_target_q = (
            (1.0 - clearance_alpha)
            * right_lift_q
            + clearance_alpha
            * right_clearance_q
        )


        data.ctrl[
            left_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL


        step_with_arm_targets(
            left_clearance_target_q,
            right_clearance_target_q,
        )

        viewer.sync()


        (
            finger_fz,
            table_fz,
            finger_normal_sum,
            finger_contact_count,
        ) = get_object_support_forces()


        if (
            first_clearance_lift_table_contact_step
            is None
            and table_fz > 0.01
        ):
            first_clearance_lift_table_contact_step = (
                clearance_step + 1
            )


    print(
        "Extra clearance lift reached."
    )

    print(
        "Table contact during clearance lift:",
        first_clearance_lift_table_contact_step,
    )

    # ========================================================
    # Pre-transport stabilization
    # ========================================================

    print(
        "\nStabilizing object at clearance height before transport..."
    )

    PRE_TRANSPORT_HOLD_STEPS = 1000

    first_pre_transport_table_contact_step = None

    for hold_step in range(
        PRE_TRANSPORT_HOLD_STEPS
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        step_with_arm_targets(
            left_clearance_q,
            right_clearance_q,
        )

        viewer.sync()

        (
            finger_fz,
            table_fz,
            finger_normal_sum,
            finger_contact_count,
        ) = get_object_support_forces()

        if (
            first_pre_transport_table_contact_step
            is None
            and table_fz > 0.01
        ):

            first_pre_transport_table_contact_step = (
                hold_step + 1
            )


    mujoco.mj_forward(
        model,
        data,
    )


    (
        object_position_before_transport,
        object_rotation_before_transport,
        object_linear_velocity_before_transport,
        object_angular_velocity_before_transport,
    ) = object_state()


    object_rpy_before_transport = (
        rotation_matrix_to_rpy(
            object_rotation_before_transport
        )
    )


    print_object_state(
        "OBJECT STATE BEFORE TRANSPORT"
    )


    (
        finger_fz_before_transport,
        table_fz_before_transport,
        finger_normal_before_transport,
        finger_contacts_before_transport,
    ) = get_object_support_forces()


    print(
        "\nPre-transport support:"
    )

    print(
        "Finger vertical support [N]:",
        finger_fz_before_transport,
    )

    print(
        "Table vertical support [N]:",
        table_fz_before_transport,
    )

    print(
        "Finger normal force sum [N]:",
        finger_normal_before_transport,
    )

    print(
        "Finger contact count:",
        finger_contacts_before_transport,
    )

    print(
        "Table contact during pre-transport hold:",
        first_pre_transport_table_contact_step,
    )


    # ========================================================
    # Cooperative rigid-body transport to goal
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )
    print(
        "STARTING COOPERATIVE GOAL TRANSPORT"
    )
    print(
        "=" * 70
    )

    print(
        "Commanded object-reference X translation [m]:",
        GOAL_TRANSPORT_DISTANCE_X,
    )

    print(
        "Commanded common world-Y rotation [deg]:",
        GOAL_COMMON_Y_ROTATION_DEG,
    )

    print(
        "IK trajectory waypoints:",
        GOAL_TRAJECTORY_WAYPOINTS,
    )

    print(
        "Physics steps per waypoint segment:",
        PHYSICS_STEPS_PER_WAYPOINT_SEGMENT,
    )

    print(
        "Total transport physics steps:",
        GOAL_TRANSPORT_STEPS,
    )

    TRANSPORT_LOG_INTERVAL = 100


    transport_initial_x = float(
        data.site_xpos[
            object_center_site_id,
            0,
        ]
    )

    transport_initial_y = float(
        data.site_xpos[
            object_center_site_id,
            1,
        ]
    )

    transport_initial_z = float(
        data.site_xpos[
            object_center_site_id,
            2,
        ]
    )


    max_transport_table_fz = 0.0

    max_transport_finger_fz = -np.inf

    min_transport_finger_fz = np.inf

    max_transport_normal_sum = -np.inf

    first_transport_table_contact_step = None
    
    max_object_orientation_error_deg = 0.0
    first_large_orientation_error_step = None

    print(
        "\n"
        "step | progress | object_x | delta_x | "
        "delta_y | delta_z | object_vx | object_vz | "
        "pitch_deg | ori_err_deg | "
        "finger_Fz | table_Fz | contacts"
    )

    print(
        "-" * 125
    )


    transport_step_index = 0


    for segment_index in range(
        GOAL_TRAJECTORY_WAYPOINTS - 1
    ):

        left_segment_start_q = (
            left_goal_trajectory_q[
                segment_index
            ]
        )

        left_segment_target_q = (
            left_goal_trajectory_q[
                segment_index + 1
            ]
        )

        right_segment_start_q = (
            right_goal_trajectory_q[
                segment_index
            ]
        )

        right_segment_target_q = (
            right_goal_trajectory_q[
                segment_index + 1
            ]
        )


        for segment_step in range(
            PHYSICS_STEPS_PER_WAYPOINT_SEGMENT
        ):

            segment_alpha = (
                segment_step + 1
            ) / PHYSICS_STEPS_PER_WAYPOINT_SEGMENT


            left_target_q = (
                (1.0 - segment_alpha)
                * left_segment_start_q
                + segment_alpha
                * left_segment_target_q
            )


            right_target_q = (
                (1.0 - segment_alpha)
                * right_segment_start_q
                + segment_alpha
                * right_segment_target_q
            )


            # Keep both grippers closed during the entire
            # cooperative goal transport.
            data.ctrl[
                left_gripper_actuator_id
            ] = CLOSE_GRIPPER_CTRL

            data.ctrl[
                right_gripper_actuator_id
            ] = CLOSE_GRIPPER_CTRL


            step_with_arm_targets(
                left_target_q,
                right_target_q,
            )

            viewer.sync()


            transport_step_index += 1


            (
                finger_fz,
                table_fz,
                finger_normal_sum,
                finger_contact_count,
            ) = get_object_support_forces()


            object_x = float(
                data.site_xpos[
                    object_center_site_id,
                    0,
                ]
            )

            object_y = float(
                data.site_xpos[
                    object_center_site_id,
                    1,
                ]
            )

            object_z = float(
                data.site_xpos[
                    object_center_site_id,
                    2,
                ]
            )


            object_vx = float(
                data.qvel[
                    object_freejoint_dof_adr
                ]
            )

            object_vz = float(
                data.qvel[
                    object_freejoint_dof_adr + 2
                ]
            )


            delta_x = (
                object_x
                - transport_initial_x
            )

            delta_y = (
                object_y
                - transport_initial_y
            )

            delta_z = (
                object_z
                - transport_initial_z
            )
            
            # ----------------------------------------------------
            # Object orientation tracking diagnostic
            # ----------------------------------------------------

            current_object_rotation = (
                data.site_xmat[
                    object_center_site_id
                ]
                .reshape(
                    3,
                    3,
                )
                .copy()
            )


            current_object_rpy = (
                rotation_matrix_to_rpy(
                    current_object_rotation
                )
            )


            current_object_pitch_deg = float(
                np.degrees(
                    current_object_rpy[1]
                )
            )


            overall_progress = (
                transport_step_index
                / GOAL_TRANSPORT_STEPS
            )


            commanded_common_angle_deg = (
                GOAL_COMMON_Y_ROTATION_DEG
                * smoothstep(
                    overall_progress
                )
            )


            desired_object_rotation = (
                rotation_y_world(
                    commanded_common_angle_deg
                )
                @ object_rotation_before_transport
            )


            current_orientation_error_deg = (
                orientation_error_deg(
                    desired_object_rotation,
                    current_object_rotation,
                )
            )


            max_object_orientation_error_deg = max(
                max_object_orientation_error_deg,
                current_orientation_error_deg,
            )


            if (
                first_large_orientation_error_step
                is None
                and current_orientation_error_deg > 2.0
            ):
                first_large_orientation_error_step = (
                    transport_step_index
                )


            max_transport_table_fz = max(
                max_transport_table_fz,
                table_fz,
            )

            max_transport_finger_fz = max(
                max_transport_finger_fz,
                finger_fz,
            )

            min_transport_finger_fz = min(
                min_transport_finger_fz,
                finger_fz,
            )

            max_transport_normal_sum = max(
                max_transport_normal_sum,
                finger_normal_sum,
            )


            if (
                first_transport_table_contact_step
                is None
                and table_fz > 0.01
            ):
                first_transport_table_contact_step = (
                    transport_step_index
                )


            if (
                transport_step_index == 1
                or transport_step_index
                % TRANSPORT_LOG_INTERVAL == 0
                or transport_step_index
                == GOAL_TRANSPORT_STEPS
            ):

                print(
                    f"{transport_step_index:4d} | "
                    f"{overall_progress:8.4f} | "
                    f"{object_x:+9.6f} | "
                    f"{delta_x:+9.6f} | "
                    f"{delta_y:+9.6f} | "
                    f"{delta_z:+9.6f} | "
                    f"{object_vx:+10.6f} | "
                    f"{object_vz:+10.6f} | "
                    f"{current_object_pitch_deg:+9.3f} | "
                    f"{current_orientation_error_deg:11.3f} | "
                    f"{finger_fz:+9.4f} | "
                    f"{table_fz:+9.4f} | "
                    f"{finger_contact_count:3d}"
                )


    # ========================================================
    # State immediately after transport
    # ========================================================

    mujoco.mj_forward(
        model,
        data,
    )


    (
        object_position_after_transport_motion,
        object_rotation_after_transport_motion,
        object_linear_velocity_after_transport_motion,
        object_angular_velocity_after_transport_motion,
    ) = object_state()


    print_object_state(
        "OBJECT STATE IMMEDIATELY AFTER TRANSPORT"
    )


    print(
        "\n"
        + "-" * 70
    )

    print(
        "DYNAMIC TRANSPORT SUMMARY"
    )

    print(
        "-" * 70
    )

    print(
        "Object X before transport:",
        object_position_before_transport[0],
    )

    print(
        "Object X after transport motion:",
        object_position_after_transport_motion[0],
    )

    print(
        "Object X displacement during motion [m]:",
        (
            object_position_after_transport_motion[0]
            - object_position_before_transport[0]
        ),
    )

    print(
        "Object Y displacement during motion [m]:",
        (
            object_position_after_transport_motion[1]
            - object_position_before_transport[1]
        ),
    )

    print(
        "Object Z displacement during motion [m]:",
        (
            object_position_after_transport_motion[2]
            - object_position_before_transport[2]
        ),
    )

    print(
        "First table contact during transport:",
        first_transport_table_contact_step,
    )
    
    print(
        "First >2 deg object orientation error step:",
        first_large_orientation_error_step,
    )

    print(
        "Maximum object orientation tracking error [deg]:",
        max_object_orientation_error_deg,
    )

    print(
        "Maximum table support during transport [N]:",
        max_transport_table_fz,
    )

    print(
        "Maximum finger vertical support [N]:",
        max_transport_finger_fz,
    )

    print(
        "Minimum finger vertical support [N]:",
        min_transport_finger_fz,
    )

    print(
        "Maximum summed finger normal force [N]:",
        max_transport_normal_sum,
    )

    print(
        "=" * 70
    )


    # ========================================================
    # Post-transport hold
    # ========================================================

    print(
        "\nHolding transported pose..."
    )


    POST_TRANSPORT_HOLD_STEPS = 2500
    POST_TRANSPORT_LOG_INTERVAL = 100


    first_post_transport_table_contact_step = None

    max_post_transport_table_fz = 0.0


    print(
        "\n"
        "step | object_x | delta_x | object_y | object_z | "
        "object_vx | object_vz | finger_Fz | table_Fz | contacts"
    )

    print(
        "-" * 120
    )


    for hold_step in range(
        POST_TRANSPORT_HOLD_STEPS
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL


        step_with_arm_targets(
            left_transport_q,
            right_transport_q,
        )

        viewer.sync()


        (
            finger_fz,
            table_fz,
            finger_normal_sum,
            finger_contact_count,
        ) = get_object_support_forces()


        object_x = float(
            data.site_xpos[
                object_center_site_id,
                0,
            ]
        )

        object_y = float(
            data.site_xpos[
                object_center_site_id,
                1,
            ]
        )

        object_z = float(
            data.site_xpos[
                object_center_site_id,
                2,
            ]
        )


        object_vx = float(
            data.qvel[
                object_freejoint_dof_adr
            ]
        )

        object_vz = float(
            data.qvel[
                object_freejoint_dof_adr + 2
            ]
        )


        delta_x = (
            object_x
            - object_position_before_transport[0]
        )


        max_post_transport_table_fz = max(
            max_post_transport_table_fz,
            table_fz,
        )


        if (
            first_post_transport_table_contact_step
            is None
            and table_fz > 0.01
        ):

            first_post_transport_table_contact_step = (
                hold_step + 1
            )


        if (
            hold_step == 0
            or (
                hold_step + 1
            ) % POST_TRANSPORT_LOG_INTERVAL == 0
            or hold_step
            == POST_TRANSPORT_HOLD_STEPS - 1
        ):

            print(
                f"{hold_step + 1:4d} | "
                f"{object_x:+9.6f} | "
                f"{delta_x:+9.6f} | "
                f"{object_y:+9.6f} | "
                f"{object_z:+9.6f} | "
                f"{object_vx:+10.6f} | "
                f"{object_vz:+10.6f} | "
                f"{finger_fz:+9.4f} | "
                f"{table_fz:+9.4f} | "
                f"{finger_contact_count:3d}"
            )


    # ========================================================
    # Final transport diagnostics
    # ========================================================

    mujoco.mj_forward(
        model,
        data,
    )


    (
        object_position_after_transport,
        object_rotation_after_transport,
        object_linear_velocity_after_transport,
        object_angular_velocity_after_transport,
    ) = object_state()


    object_rpy_after_transport = (
        rotation_matrix_to_rpy(
            object_rotation_after_transport
        )
    )


    object_translation = (
        object_position_after_transport
        - object_position_before_transport
    )
    
    x_completion_ratio = (
        object_translation[0]
        / GOAL_TRANSPORT_DISTANCE_X
    )

    x_completion_percentage = (
        x_completion_ratio
        * 100.0
    )

    object_rpy_change = (
        object_rpy_after_transport
        - object_rpy_before_transport
    )


    print_object_state(
        "FINAL OBJECT STATE AFTER TRANSPORT HOLD"
    )


    print_object_contacts(
        "FINAL CONTACTS AFTER TRANSPORT HOLD"
    )


    print_finger_joint_positions()


    print_gripper_actuator_state(
        "FINAL GRIPPER ACTUATOR STATE"
    )


    print_gripper_transmission_state(
        "FINAL GRIPPER TRANSMISSION STATE"
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "COOPERATIVE TRANSPORT RESULT"
    )

    print(
        "=" * 70
    )


    print(
        "Object position before transport:",
        object_position_before_transport,
    )

    print(
        "Object position after transport:",
        object_position_after_transport,
    )

    print(
        "Object translation [m]:",
        object_translation,
    )

    print(
        "Commanded X translation [m]:",
        GOAL_TRANSPORT_DISTANCE_X,
    )

    print(
        "Commanded common world-Y rotation [deg]:",
        GOAL_COMMON_Y_ROTATION_DEG,
    )

    print(
        "Actual object X translation [m]:",
        object_translation[0],
    )
    
    print(
        "X transport completion [%]:",
        x_completion_percentage,
    )

    print(
        "Object Y drift [m]:",
        object_translation[1],
    )

    print(
        "Object Z change [m]:",
        object_translation[2],
    )


    print(
        "\nObject RPY before transport [deg]:",
        np.degrees(
            object_rpy_before_transport
        ),
    )

    print(
        "Object RPY after transport [deg]:",
        np.degrees(
            object_rpy_after_transport
        ),
    )

    print(
        "Object RPY change [deg]:",
        np.degrees(
            object_rpy_change
        ),
    )


    print(
        "\nFinal object linear velocity [m/s]:",
        object_linear_velocity_after_transport,
    )

    print(
        "Final object angular velocity [rad/s]:",
        object_angular_velocity_after_transport,
    )


    print(
        "\nFirst table contact during transport:",
        first_transport_table_contact_step,
    )

    print(
        "First table contact during final hold:",
        first_post_transport_table_contact_step,
    )

    print(
        "Maximum table support during transport [N]:",
        max_transport_table_fz,
    )

    print(
        "Maximum table support during final hold [N]:",
        max_post_transport_table_fz,
    )


    # --------------------------------------------------------
    # Arm tracking diagnostics
    # --------------------------------------------------------

    left_actual_q = get_arm_qpos(
        left_qpos_ids
    )

    right_actual_q = get_arm_qpos(
        right_qpos_ids
    )


    left_q_error_norm = np.linalg.norm(
        left_transport_q
        - left_actual_q
    )

    right_q_error_norm = np.linalg.norm(
        right_transport_q
        - right_actual_q
    )


    print(
        "\nLEFT ARM"
    )

    print(
        "Commanded transport q:",
        left_transport_q,
    )

    print(
        "Actual q:",
        left_actual_q,
    )

    print(
        "q error norm:",
        left_q_error_norm,
    )


    print(
        "\nRIGHT ARM"
    )

    print(
        "Commanded transport q:",
        right_transport_q,
    )

    print(
        "Actual q:",
        right_actual_q,
    )

    print(
        "q error norm:",
        right_q_error_norm,
    )


    # ========================================================
    # First transport acceptance criteria
    # ========================================================

    MINIMUM_X_COMPLETION_RATIO = 0.90

    minimum_required_x_translation = (
        GOAL_TRANSPORT_DISTANCE_X
        * MINIMUM_X_COMPLETION_RATIO
    )

    x_translation_pass = (
        object_translation[0]
        >= minimum_required_x_translation
    )

    y_drift_pass = (
        abs(
            object_translation[1]
        ) <= 0.010
    )

    no_transport_table_contact_pass = (
        first_transport_table_contact_step
        is None
    )

    no_hold_table_contact_pass = (
        first_post_transport_table_contact_step
        is None
    )

    linear_velocity_pass = (
        np.linalg.norm(
            object_linear_velocity_after_transport
        ) <= 0.01
    )

    angular_velocity_pass = (
        np.linalg.norm(
            object_angular_velocity_after_transport
        ) <= 0.10
    )
    
    # The goal trajectory intentionally tilts the rigid grasp
    # approximately -9 degrees about world Y.
    #
    # Therefore "orientation stayed within 5 degrees of start"
    # is no longer a valid criterion.
    #
    # For this first dynamic goal test, verify that the dominant
    # object pitch change is reasonably close to the commanded
    # common rotation.

    object_rpy_change_deg = np.degrees(
        object_rpy_change
    )


    object_pitch_change_deg = (
        object_rpy_change_deg[1]
    )


    OBJECT_PITCH_TOLERANCE_DEG = 3.0


    orientation_pass = (
        abs(
            object_pitch_change_deg
            - GOAL_COMMON_Y_ROTATION_DEG
        )
        <= OBJECT_PITCH_TOLERANCE_DEG
    )

    transport_pass = all([
        x_translation_pass,
        y_drift_pass,
        no_transport_table_contact_pass,
        no_hold_table_contact_pass,
        linear_velocity_pass,
        angular_velocity_pass,
        orientation_pass,
    ])


    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRANSPORT ACCEPTANCE CHECK"
    )

    print(
        "=" * 70
    )

    print(
        "X translation >=",
        minimum_required_x_translation,
        "m:",
        x_translation_pass,
    )

    print(
        "|Y drift| <= 0.010 m:",
        y_drift_pass,
    )

    print(
        "No table contact during transport:",
        no_transport_table_contact_pass,
    )

    print(
        "No table contact during final hold:",
        no_hold_table_contact_pass,
    )

    print(
        "Final linear velocity norm <= 0.01 m/s:",
        linear_velocity_pass,
    )

    print(
        "Final angular velocity norm <= 0.10 rad/s:",
        angular_velocity_pass,
    )

    print(
        "Object pitch change [deg]:",
        object_pitch_change_deg,
    )

    print(
        "Commanded common world-Y rotation [deg]:",
        GOAL_COMMON_Y_ROTATION_DEG,
    )

    print(
        "Pitch error <=",
        OBJECT_PITCH_TOLERANCE_DEG,
        "deg:",
        orientation_pass,
    )


    if transport_pass:

        print(
            "\nRESULT: PASS"
        )

        print(
            "Stable cooperative goal transport validated "
            f"for {GOAL_TRANSPORT_DISTANCE_X * 100:.1f} cm "
            f"with {GOAL_COMMON_Y_ROTATION_DEG:.1f} deg "
            "common world-Y rotation."
        )

    else:

        print(
            "\nRESULT: FAIL"
        )

        print(
            "Inspect the failed acceptance criteria above."
        )


    print(
        "=" * 70
    )


    # ========================================================
    # Keep viewer open briefly
    # ========================================================

    print(
        "\nHolding final transported configuration..."
    )


    for _ in range(
        2000
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        step_with_arm_targets(
            left_transport_q,
            right_transport_q,
        )

        viewer.sync()

        time.sleep(
            0.001
        )
