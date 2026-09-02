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
# Cooperative flat transport / placement configuration
# ============================================================

# The benchmark goal was calibrated to X = 0.420 m after
# workspace-feasibility analysis.
#
# The flat near-table reachability scan showed:
#
#   X = 0.420 m:
#       position error    ~ 0.100 mm
#       orientation error ~ 0.005 deg
#
#   X = 0.422 m:
#       position error    ~ 0.742 mm
#
#   X = 0.423 m:
#       position error    ~ 1.158 mm -> FAIL
#
# Therefore X = 0.420 m provides a useful margin from the
# measured bilateral flat-placement IK boundary.

PLACEMENT_X = 0.404

# Maximum comfortable suspended flat-transport target.
#
# Previous fixed-orientation reachability tests showed that
# the lifted pose at Z = 0.580 is reliably reachable around
# X = 0.400, while accuracy degrades strongly beyond ~0.404.

TRANSPORT_X = 0.380


# Suspended horizontal transport is performed at the already
# validated lift height.
TRANSPORT_Z = 0.580


LEFT_TRANSPORT_POSITION = np.array([
    TRANSPORT_X,
    0.145,
    TRANSPORT_Z,
])

RIGHT_TRANSPORT_POSITION = np.array([
    TRANSPORT_X,
    -0.145,
    TRANSPORT_Z,
])


# Flat near-table placement reference.
#
# From the validated suspended grasp:
#
# object_center_z - EE_reference_z ~= -0.006015 m
#
# A flat 0.12 m-high object resting on the 0.50 m table has
# object-center Z ~= 0.560 m.
#
# Therefore the estimated EE reference height for table
# contact is:
#
# 0.560 + 0.006015 ~= 0.566015 m
#
# The reachability scan validated Z = 0.566 m.

PLACEMENT_Z = 0.566


LEFT_PLACEMENT_POSITION = np.array([
    PLACEMENT_X,
    0.145,
    PLACEMENT_Z,
])

RIGHT_PLACEMENT_POSITION = np.array([
    PLACEMENT_X,
    -0.145,
    PLACEMENT_Z,
])


# Cartesian IK waypoint counts.
#
# Horizontal transport:
#   x = 0.000 -> 0.420 m
#
# Placement:
#   z = 0.580 -> 0.566 m

TRANSPORT_TRAJECTORY_WAYPOINTS = 43

PLACEMENT_TRAJECTORY_WAYPOINTS = 15


# Physics interpolation between consecutive IK waypoints.

PHYSICS_STEPS_PER_WAYPOINT_SEGMENT = 100


TRANSPORT_STEPS = (
    (TRANSPORT_TRAJECTORY_WAYPOINTS - 1)
    * PHYSICS_STEPS_PER_WAYPOINT_SEGMENT
)


PLACEMENT_STEPS = (
    (PLACEMENT_TRAJECTORY_WAYPOINTS - 1)
    * PHYSICS_STEPS_PER_WAYPOINT_SEGMENT
)


# IK validation tolerances.

IK_POSITION_TOLERANCE_M = 1.0e-3

IK_ORIENTATION_TOLERANCE_DEG = 1.0

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
# Build flat horizontal transport trajectory
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "BUILDING FLAT HORIZONTAL TRANSPORT TRAJECTORY"
)

print(
    "=" * 70
)


left_transport_trajectory_q = [
    left_lift_q.copy()
]

right_transport_trajectory_q = [
    right_lift_q.copy()
]


previous_left_q = (
    left_lift_q.copy()
)

previous_right_q = (
    right_lift_q.copy()
)


transport_trajectory_valid = True


for waypoint_index in range(
    1,
    TRANSPORT_TRAJECTORY_WAYPOINTS,
):

    raw_alpha = (
        waypoint_index
        / (
            TRANSPORT_TRAJECTORY_WAYPOINTS
            - 1
        )
    )


    trajectory_alpha = smoothstep(
        raw_alpha
    )


    current_x = (
        TRANSPORT_X
        * trajectory_alpha
    )


    left_target_position = np.array([
        current_x,
        0.145,
        TRANSPORT_Z,
    ])


    right_target_position = np.array([
        current_x,
        -0.145,
        TRANSPORT_Z,
    ])


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
        target_rotation=left_target_rotation,
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
        target_rotation=right_target_rotation,
        initial_q=previous_right_q,
    )


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
            left_target_rotation,
            left_achieved_rotation,
        )
    )


    right_orientation_error = (
        orientation_error_deg(
            right_target_rotation,
            right_achieved_rotation,
        )
    )


    max_position_error = max(
        left_position_error,
        right_position_error,
    )


    max_orientation_error = max(
        left_orientation_error,
        right_orientation_error,
    )


    waypoint_pass = all([
        max_position_error
        <= IK_POSITION_TOLERANCE_M,

        max_orientation_error
        <= IK_ORIENTATION_TOLERANCE_DEG,
    ])


    print(
        f"Transport waypoint "
        f"{waypoint_index:02d}/"
        f"{TRANSPORT_TRAJECTORY_WAYPOINTS - 1:02d}"
        f" | x={current_x:.4f} m"
        f" | pos={max_position_error * 1000.0:.3f} mm"
        f" | ori={max_orientation_error:.4f} deg"
        f" | PASS={waypoint_pass}"
    )


    if not waypoint_pass:

        transport_trajectory_valid = False

        break


    left_transport_trajectory_q.append(
        left_waypoint_q.copy()
    )

    right_transport_trajectory_q.append(
        right_waypoint_q.copy()
    )


    previous_left_q = (
        left_waypoint_q.copy()
    )

    previous_right_q = (
        right_waypoint_q.copy()
    )


if not transport_trajectory_valid:

    raise RuntimeError(
        "Flat horizontal transport trajectory failed "
        "IK validation. Dynamic execution was NOT started."
    )


if (
    len(left_transport_trajectory_q)
    != TRANSPORT_TRAJECTORY_WAYPOINTS
    or len(right_transport_trajectory_q)
    != TRANSPORT_TRAJECTORY_WAYPOINTS
):

    raise RuntimeError(
        "Flat transport trajectory contains an unexpected "
        "number of waypoints."
    )


left_transport_q = (
    left_transport_trajectory_q[-1].copy()
)

right_transport_q = (
    right_transport_trajectory_q[-1].copy()
)


print(
    "\nFlat horizontal transport trajectory: PASS"
)

print(
    "Final transport target X [m]:",
    PLACEMENT_X,
)


# ============================================================
# Build vertical flat-placement trajectory
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "BUILDING DIAGONAL FLAT-PLACEMENT TRAJECTORY"
)

print(
    "=" * 70
)


left_placement_trajectory_q = [
    left_transport_q.copy()
]

right_placement_trajectory_q = [
    right_transport_q.copy()
]


previous_left_q = (
    left_transport_q.copy()
)

previous_right_q = (
    right_transport_q.copy()
)


placement_trajectory_valid = True


for waypoint_index in range(
    1,
    PLACEMENT_TRAJECTORY_WAYPOINTS,
):

    raw_alpha = (
        waypoint_index
        / (
            PLACEMENT_TRAJECTORY_WAYPOINTS
            - 1
        )
    )


    trajectory_alpha = smoothstep(
        raw_alpha
    )

    current_x = (
        TRANSPORT_X
        + (
            PLACEMENT_X
            - TRANSPORT_X
        )
        * trajectory_alpha
    )

    current_z = (
        TRANSPORT_Z
        + (
            PLACEMENT_Z
            - TRANSPORT_Z
        )
        * trajectory_alpha
    )


    left_target_position = np.array([
        current_x,
        0.145,
        current_z,
    ])

    right_target_position = np.array([
        current_x,
        -0.145,
        current_z,
    ])


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
        target_rotation=left_target_rotation,
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
        target_rotation=right_target_rotation,
        initial_q=previous_right_q,
    )


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
            left_target_rotation,
            left_achieved_rotation,
        )
    )


    right_orientation_error = (
        orientation_error_deg(
            right_target_rotation,
            right_achieved_rotation,
        )
    )


    max_position_error = max(
        left_position_error,
        right_position_error,
    )


    max_orientation_error = max(
        left_orientation_error,
        right_orientation_error,
    )


    waypoint_pass = all([
        max_position_error
        <= IK_POSITION_TOLERANCE_M,

        max_orientation_error
        <= IK_ORIENTATION_TOLERANCE_DEG,
    ])


    print(
        f"Placement waypoint "
        f"{waypoint_index:02d}/"
        f"{PLACEMENT_TRAJECTORY_WAYPOINTS - 1:02d}"
        f" | x={current_x:.4f} m"
        f" | z={current_z:.6f} m"
        f" | pos={max_position_error * 1000.0:.3f} mm"
        f" | ori={max_orientation_error:.4f} deg"
        f" | PASS={waypoint_pass}"
    )


    if not waypoint_pass:

        placement_trajectory_valid = False

        break


    left_placement_trajectory_q.append(
        left_waypoint_q.copy()
    )

    right_placement_trajectory_q.append(
        right_waypoint_q.copy()
    )


    previous_left_q = (
        left_waypoint_q.copy()
    )

    previous_right_q = (
        right_waypoint_q.copy()
    )


if not placement_trajectory_valid:

    raise RuntimeError(
        "Vertical flat-placement trajectory failed "
        "IK validation. Dynamic execution was NOT started."
    )


if (
    len(left_placement_trajectory_q)
    != PLACEMENT_TRAJECTORY_WAYPOINTS
    or len(right_placement_trajectory_q)
    != PLACEMENT_TRAJECTORY_WAYPOINTS
):

    raise RuntimeError(
        "Placement trajectory contains an unexpected "
        "number of waypoints."
    )


left_placement_q = (
    left_placement_trajectory_q[-1].copy()
)

right_placement_q = (
    right_placement_trajectory_q[-1].copy()
)


print(
    "\nDiagonal flat-placement trajectory: PASS"
)

print(
    "Final placement target:"
)

print(
    "  X [m]:",
    PLACEMENT_X,
)

print(
    "  EE reference Z [m]:",
    PLACEMENT_Z,
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
    # Pre-transport stabilization at validated lift pose
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PRE-TRANSPORT STABILIZATION"
    )

    print(
        "=" * 70
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
            left_lift_q,
            right_lift_q,
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


    (
        finger_fz_before_transport,
        table_fz_before_transport,
        finger_normal_before_transport,
        finger_contacts_before_transport,
    ) = get_object_support_forces()


    print_object_state(
        "OBJECT STATE BEFORE FLAT TRANSPORT"
    )


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
    # Flat horizontal cooperative transport
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STARTING FLAT HORIZONTAL COOPERATIVE TRANSPORT"
    )

    print(
        "=" * 70
    )

    print(
        "Target suspended transport X [m]:",
        TRANSPORT_X,
    )

    print(
        "Transport EE-reference Z [m]:",
        TRANSPORT_Z,
    )

    print(
        "IK trajectory waypoints:",
        TRANSPORT_TRAJECTORY_WAYPOINTS,
    )

    print(
        "Physics steps per waypoint segment:",
        PHYSICS_STEPS_PER_WAYPOINT_SEGMENT,
    )

    print(
        "Total transport physics steps:",
        TRANSPORT_STEPS,
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

    max_transport_orientation_change_deg = 0.0


    print(
        "\n"
        "step | progress | object_x | delta_x | "
        "delta_y | delta_z | object_vx | object_vz | "
        "ori_change_deg | finger_Fz | table_Fz | contacts"
    )

    print(
        "-" * 130
    )


    transport_step_index = 0


    for segment_index in range(
        TRANSPORT_TRAJECTORY_WAYPOINTS - 1
    ):

        left_segment_start_q = (
            left_transport_trajectory_q[
                segment_index
            ]
        )

        left_segment_target_q = (
            left_transport_trajectory_q[
                segment_index + 1
            ]
        )

        right_segment_start_q = (
            right_transport_trajectory_q[
                segment_index
            ]
        )

        right_segment_target_q = (
            right_transport_trajectory_q[
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


            current_orientation_change_deg = (
                orientation_error_deg(
                    object_rotation_before_transport,
                    current_object_rotation,
                )
            )


            max_transport_orientation_change_deg = max(
                max_transport_orientation_change_deg,
                current_orientation_change_deg,
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
                or (
                    transport_step_index
                    % TRANSPORT_LOG_INTERVAL
                    == 0
                )
                or transport_step_index
                == TRANSPORT_STEPS
            ):

                progress = (
                    transport_step_index
                    / TRANSPORT_STEPS
                )


                print(
                    f"{transport_step_index:4d} | "
                    f"{progress:8.4f} | "
                    f"{object_x:+9.6f} | "
                    f"{delta_x:+9.6f} | "
                    f"{delta_y:+9.6f} | "
                    f"{delta_z:+9.6f} | "
                    f"{object_vx:+10.6f} | "
                    f"{object_vz:+10.6f} | "
                    f"{current_orientation_change_deg:13.6f} | "
                    f"{finger_fz:+9.4f} | "
                    f"{table_fz:+9.4f} | "
                    f"{finger_contact_count:3d}"
                )


    # ========================================================
    # Hold after horizontal transport
    # ========================================================

    print(
        "\nHorizontal trajectory complete."
    )

    print(
        "Stabilizing at transport destination..."
    )


    POST_TRANSPORT_HOLD_STEPS = 1500

    POST_TRANSPORT_LOG_INTERVAL = 250

    first_post_transport_table_contact_step = None

    max_post_transport_table_fz = 0.0


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


            print(
                f"hold {hold_step + 1:4d}"
                f" | x={object_x:+.6f}"
                f" | y={object_y:+.6f}"
                f" | z={object_z:+.6f}"
                f" | finger_Fz={finger_fz:+.4f}"
                f" | table_Fz={table_fz:+.4f}"
                f" | contacts={finger_contact_count}"
            )


    # ========================================================
    # Horizontal transport result
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


    object_transport_translation = (
        object_position_after_transport
        - object_position_before_transport
    )


    transport_orientation_change_deg = (
        orientation_error_deg(
            object_rotation_before_transport,
            object_rotation_after_transport,
        )
    )


    expected_transport_x = (
        TRANSPORT_X
        - object_position_before_transport[0]
    )


    TRANSPORT_X_TOLERANCE_M = 0.010

    TRANSPORT_Y_TOLERANCE_M = 0.010

    TRANSPORT_Z_TOLERANCE_M = 0.010


    transport_x_pass = (
        abs(
            object_position_after_transport[0]
            - TRANSPORT_X
        )
        <= TRANSPORT_X_TOLERANCE_M
    )


    transport_y_pass = (
        abs(
            object_position_after_transport[1]
        )
        <= TRANSPORT_Y_TOLERANCE_M
    )


    transport_z_pass = (
        abs(
            object_position_after_transport[2]
            - object_position_before_transport[2]
        )
        <= TRANSPORT_Z_TOLERANCE_M
    )


    no_transport_table_contact_pass = (
        first_transport_table_contact_step
        is None
    )


    no_transport_hold_table_contact_pass = (
        first_post_transport_table_contact_step
        is None
    )


    transport_linear_velocity_pass = (
        np.linalg.norm(
            object_linear_velocity_after_transport
        )
        <= 0.01
    )


    transport_angular_velocity_pass = (
        np.linalg.norm(
            object_angular_velocity_after_transport
        )
        <= 0.10
    )


    transport_orientation_pass = (
        transport_orientation_change_deg
        <= 2.0
    )


    horizontal_transport_pass = all([
        transport_x_pass,
        transport_y_pass,
        transport_z_pass,
        no_transport_table_contact_pass,
        no_transport_hold_table_contact_pass,
        transport_linear_velocity_pass,
        transport_angular_velocity_pass,
        transport_orientation_pass,
    ])


    print(
        "\n"
        + "=" * 70
    )

    print(
        "HORIZONTAL TRANSPORT RESULT"
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
        object_transport_translation,
    )

    print(
        "Expected destination X [m]:",
        TRANSPORT_X,
    )

    print(
        "Actual destination X [m]:",
        object_position_after_transport[0],
    )

    print(
        "Expected transport X displacement [m]:",
        expected_transport_x,
    )

    print(
        "Orientation change [deg]:",
        transport_orientation_change_deg,
    )

    print(
        "Maximum orientation change during transport [deg]:",
        max_transport_orientation_change_deg,
    )

    print(
        "First table contact during transport:",
        first_transport_table_contact_step,
    )

    print(
        "First table contact during transport hold:",
        first_post_transport_table_contact_step,
    )

    print(
        "Maximum table support during transport [N]:",
        max_transport_table_fz,
    )

    print(
        "Maximum table support during transport hold [N]:",
        max_post_transport_table_fz,
    )

    print(
        "Final linear velocity [m/s]:",
        object_linear_velocity_after_transport,
    )

    print(
        "Final angular velocity [rad/s]:",
        object_angular_velocity_after_transport,
    )


    print(
        "\nHorizontal transport checks:"
    )

    print(
        f"|object_x - {TRANSPORT_X:.3f}| <= 0.010 m:",
        transport_x_pass,
    )

    print(
        "|object_y| <= 0.010 m:",
        transport_y_pass,
    )

    print(
        "|transport Z change| <= 0.010 m:",
        transport_z_pass,
    )

    print(
        "No table contact during horizontal motion:",
        no_transport_table_contact_pass,
    )

    print(
        "No table contact during horizontal hold:",
        no_transport_hold_table_contact_pass,
    )

    print(
        "Final linear velocity norm <= 0.01 m/s:",
        transport_linear_velocity_pass,
    )

    print(
        "Final angular velocity norm <= 0.10 rad/s:",
        transport_angular_velocity_pass,
    )

    print(
        "Orientation change <= 2 deg:",
        transport_orientation_pass,
    )


    if not horizontal_transport_pass:

        raise RuntimeError(
            "Flat horizontal cooperative transport FAILED. "
            "Placement lowering was NOT started."
        )


    print(
        "\nHORIZONTAL TRANSPORT: PASS"
    )


    # ========================================================
    # Vertical lowering for flat placement
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STARTING DIAGONAL FLAT PLACEMENT"
    )

    print(
        "=" * 70
    )

    print(
        "Placement start X [m]:",
        TRANSPORT_X,
    )

    print(
        "Placement final X [m]:",
        PLACEMENT_X,
    )

    print(
        "Starting EE reference Z [m]:",
        TRANSPORT_Z,
    )

    print(
        "Final EE reference Z [m]:",
        PLACEMENT_Z,
    )

    print(
        "Placement trajectory waypoints:",
        PLACEMENT_TRAJECTORY_WAYPOINTS,
    )

    print(
        "Total placement physics steps:",
        PLACEMENT_STEPS,
    )


    PLACEMENT_LOG_INTERVAL = 100


    (
        object_position_before_placement,
        object_rotation_before_placement,
        object_linear_velocity_before_placement,
        object_angular_velocity_before_placement,
    ) = object_state()


    first_placement_table_contact_step = None

    max_placement_table_fz = 0.0

    max_placement_orientation_change_deg = 0.0

    placement_step_index = 0


    print(
        "\n"
        "step | progress | object_x | object_y | object_z | "
        "delta_z | object_vz | ori_change_deg | "
        "finger_Fz | table_Fz | contacts"
    )

    print(
        "-" * 125
    )


    for segment_index in range(
        PLACEMENT_TRAJECTORY_WAYPOINTS - 1
    ):

        left_segment_start_q = (
            left_placement_trajectory_q[
                segment_index
            ]
        )

        left_segment_target_q = (
            left_placement_trajectory_q[
                segment_index + 1
            ]
        )

        right_segment_start_q = (
            right_placement_trajectory_q[
                segment_index
            ]
        )

        right_segment_target_q = (
            right_placement_trajectory_q[
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


            placement_step_index += 1


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


            object_vz = float(
                data.qvel[
                    object_freejoint_dof_adr + 2
                ]
            )


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


            placement_orientation_change_deg = (
                orientation_error_deg(
                    object_rotation_before_placement,
                    current_object_rotation,
                )
            )


            max_placement_orientation_change_deg = max(
                max_placement_orientation_change_deg,
                placement_orientation_change_deg,
            )


            max_placement_table_fz = max(
                max_placement_table_fz,
                table_fz,
            )


            if (
                first_placement_table_contact_step
                is None
                and table_fz > 0.01
            ):

                first_placement_table_contact_step = (
                    placement_step_index
                )


            if (
                placement_step_index == 1
                or (
                    placement_step_index
                    % PLACEMENT_LOG_INTERVAL
                    == 0
                )
                or placement_step_index
                == PLACEMENT_STEPS
            ):

                placement_progress = (
                    placement_step_index
                    / PLACEMENT_STEPS
                )


                print(
                    f"{placement_step_index:4d} | "
                    f"{placement_progress:8.4f} | "
                    f"{object_x:+9.6f} | "
                    f"{object_y:+9.6f} | "
                    f"{object_z:+9.6f} | "
                    f"{object_z - object_position_before_placement[2]:+9.6f} | "
                    f"{object_vz:+10.6f} | "
                    f"{placement_orientation_change_deg:13.6f} | "
                    f"{finger_fz:+9.4f} | "
                    f"{table_fz:+9.4f} | "
                    f"{finger_contact_count:3d}"
                )


    # ========================================================
    # Placement stabilization while still grasping
    # ========================================================

    print(
        "\nPlacement trajectory complete."
    )

    print(
        "Holding flat placement with grippers still closed..."
    )


    PLACEMENT_HOLD_STEPS = 2000

    PLACEMENT_HOLD_LOG_INTERVAL = 250


    first_placement_hold_table_contact_step = None

    max_placement_hold_table_fz = 0.0


    for hold_step in range(
        PLACEMENT_HOLD_STEPS
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL


        step_with_arm_targets(
            left_placement_q,
            right_placement_q,
        )

        viewer.sync()


        (
            finger_fz,
            table_fz,
            finger_normal_sum,
            finger_contact_count,
        ) = get_object_support_forces()


        max_placement_hold_table_fz = max(
            max_placement_hold_table_fz,
            table_fz,
        )


        if (
            first_placement_hold_table_contact_step
            is None
            and table_fz > 0.01
        ):

            first_placement_hold_table_contact_step = (
                hold_step + 1
            )


        if (
            hold_step == 0
            or (
                hold_step + 1
            ) % PLACEMENT_HOLD_LOG_INTERVAL == 0
            or hold_step
            == PLACEMENT_HOLD_STEPS - 1
        ):

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


            object_vz = float(
                data.qvel[
                    object_freejoint_dof_adr + 2
                ]
            )


            print(
                f"hold {hold_step + 1:4d}"
                f" | x={object_x:+.6f}"
                f" | y={object_y:+.6f}"
                f" | z={object_z:+.6f}"
                f" | vz={object_vz:+.6f}"
                f" | finger_Fz={finger_fz:+.4f}"
                f" | table_Fz={table_fz:+.4f}"
                f" | contacts={finger_contact_count}"
            )


    # ========================================================
    # Final placement diagnostics
    # ========================================================

    mujoco.mj_forward(
        model,
        data,
    )


    (
        object_position_after_placement,
        object_rotation_after_placement,
        object_linear_velocity_after_placement,
        object_angular_velocity_after_placement,
    ) = object_state()


    object_rpy_after_placement = (
        rotation_matrix_to_rpy(
            object_rotation_after_placement
        )
    )


    final_placement_orientation_change_deg = (
        orientation_error_deg(
            object_rotation_before_placement,
            object_rotation_after_placement,
        )
    )


    (
        final_finger_fz,
        final_table_fz,
        final_finger_normal_sum,
        final_finger_contact_count,
    ) = get_object_support_forces()


    print_object_state(
        "FINAL OBJECT STATE AFTER PLACEMENT HOLD"
    )


    print_object_contacts(
        "FINAL CONTACTS AFTER PLACEMENT HOLD"
    )


    print_finger_joint_positions()


    print_gripper_actuator_state(
        "PLACEMENT-HOLD GRIPPER ACTUATOR STATE"
    )


    # ========================================================
    # Placement acceptance criteria
    # ========================================================

    # Goal:
    #
    # center = (0.420, 0)
    # half-size X = 0.220 m
    # half-size Y = 0.150 m
    #
    # Object:
    #
    # half-size X = 0.200 m
    # half-size Y = 0.120 m
    #
    # For a nearly flat/aligned object, full-footprint
    # containment therefore requires:
    #
    # |object_x - goal_x| <= 0.020 m
    # |object_y - goal_y| <= 0.030 m

    GOAL_CENTER_X = 0.420

    GOAL_CENTER_Y = 0.0

    GOAL_HALF_X = 0.220

    GOAL_HALF_Y = 0.150

    OBJECT_HALF_X = 0.200

    OBJECT_HALF_Y = 0.120


    GOAL_X_CENTER_TOLERANCE = (
        GOAL_HALF_X
        - OBJECT_HALF_X
    )

    GOAL_Y_CENTER_TOLERANCE = (
        GOAL_HALF_Y
        - OBJECT_HALF_Y
    )


    goal_x_pass = (
        abs(
            object_position_after_placement[0]
            - GOAL_CENTER_X
        )
        <= GOAL_X_CENTER_TOLERANCE
    )


    goal_y_pass = (
        abs(
            object_position_after_placement[1]
            - GOAL_CENTER_Y
        )
        <= GOAL_Y_CENTER_TOLERANCE
    )


    table_contact_pass = (
        first_placement_table_contact_step
        is not None
        or first_placement_hold_table_contact_step
        is not None
    )


    final_table_support_pass = (
        final_table_fz > 1.0
    )


    placement_linear_velocity_pass = (
        np.linalg.norm(
            object_linear_velocity_after_placement
        )
        <= 0.01
    )


    placement_angular_velocity_pass = (
        np.linalg.norm(
            object_angular_velocity_after_placement
        )
        <= 0.10
    )


    placement_orientation_pass = (
        final_placement_orientation_change_deg
        <= 2.0
    )


    placement_pass = all([
        goal_x_pass,
        goal_y_pass,
        table_contact_pass,
        final_table_support_pass,
        placement_linear_velocity_pass,
        placement_angular_velocity_pass,
        placement_orientation_pass,
    ])


    print(
        "\n"
        + "=" * 70
    )

    print(
        "FLAT PLACEMENT RESULT"
    )

    print(
        "=" * 70
    )


    print(
        "Final object position:",
        object_position_after_placement,
    )

    print(
        "Final object RPY [deg]:",
        np.degrees(
            object_rpy_after_placement
        ),
    )

    print(
        "Final linear velocity [m/s]:",
        object_linear_velocity_after_placement,
    )

    print(
        "Final angular velocity [rad/s]:",
        object_angular_velocity_after_placement,
    )


    print(
        "\nPlacement contact diagnostics:"
    )

    print(
        "First table contact during lowering:",
        first_placement_table_contact_step,
    )

    print(
        "First table contact during placement hold:",
        first_placement_hold_table_contact_step,
    )

    print(
        "Maximum table support during lowering [N]:",
        max_placement_table_fz,
    )

    print(
        "Maximum table support during placement hold [N]:",
        max_placement_hold_table_fz,
    )

    print(
        "Final table vertical support [N]:",
        final_table_fz,
    )

    print(
        "Final finger vertical support [N]:",
        final_finger_fz,
    )

    print(
        "Final finger normal force sum [N]:",
        final_finger_normal_sum,
    )

    print(
        "Final finger contact count:",
        final_finger_contact_count,
    )


    print(
        "\nPlacement checks:"
    )

    print(
        "|object_x - goal_x| <=",
        GOAL_X_CENTER_TOLERANCE,
        "m:",
        goal_x_pass,
    )

    print(
        "|object_y - goal_y| <=",
        GOAL_Y_CENTER_TOLERANCE,
        "m:",
        goal_y_pass,
    )

    print(
        "Table contact occurred during placement:",
        table_contact_pass,
    )

    print(
        "Final table support > 1 N:",
        final_table_support_pass,
    )

    print(
        "Final linear velocity norm <= 0.01 m/s:",
        placement_linear_velocity_pass,
    )

    print(
        "Final angular velocity norm <= 0.10 rad/s:",
        placement_angular_velocity_pass,
    )

    print(
        "Final orientation change <= 2 deg:",
        placement_orientation_pass,
    )


    if placement_pass:

        print(
            "\nRESULT: PASS"
        )

        print(
            "Stable flat cooperative placement validated "
            "at the calibrated x = 0.420 m goal."
        )

    else:

        print(
            "\nRESULT: FAIL"
        )

        print(
            "Do NOT proceed to gripper release yet. "
            "Inspect the failed placement criterion above."
        )


    print(
        "=" * 70
    )

    # ========================================================
    # Keep viewer open briefly
    # ========================================================

    print(
        "\nHolding final placed configuration..."
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
            left_placement_q,
            right_placement_q,
        )

        viewer.sync()

        time.sleep(
            0.001
        )
