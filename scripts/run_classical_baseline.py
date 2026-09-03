from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.controllers.ik_controller import (
    solve_pose_ik,
)

from src.controllers.classical_controller import (
    ClassicalController,
    ClassicalPhase,
)


# ============================================================
# Model
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "assets/models/dual_panda/transport_scene.xml"
)


# ============================================================
# Validated task configuration
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
    0.560,
])

RIGHT_PREGRASP_POSITION = np.array([
    0.0,
    -0.185,
    0.560,
])


LEFT_GRASP_POSITION = np.array([
    0.0,
    0.145,
    0.560,
])

RIGHT_GRASP_POSITION = np.array([
    0.0,
    -0.145,
    0.560,
])


LEFT_LIFT_POSITION = np.array([
    0.0,
    0.145,
    0.580,
])

RIGHT_LIFT_POSITION = np.array([
    0.0,
    -0.145,
    0.580,
])


# ------------------------------------------------------------
# Horizontal suspended transport
# ------------------------------------------------------------

TRANSPORT_X = 0.380
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


# ------------------------------------------------------------
# Flat placement
# ------------------------------------------------------------

PLACEMENT_X = 0.408
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


# ------------------------------------------------------------
# Outward retreat
# ------------------------------------------------------------

RETREAT_Y = 0.185


LEFT_RETREAT_POSITION = np.array([
    PLACEMENT_X,
    RETREAT_Y,
    PLACEMENT_Z,
])

RIGHT_RETREAT_POSITION = np.array([
    PLACEMENT_X,
    -RETREAT_Y,
    PLACEMENT_Z,
])


# ============================================================
# Trajectory resolution
# ============================================================

TRANSPORT_TRAJECTORY_WAYPOINTS = 43

PLACEMENT_TRAJECTORY_WAYPOINTS = 15

RETREAT_TRAJECTORY_WAYPOINTS = 21


PHYSICS_STEPS_PER_WAYPOINT_SEGMENT = 100

RETREAT_PHYSICS_STEPS_PER_WAYPOINT_SEGMENT = 100


# ============================================================
# Phase durations
# ============================================================

INITIAL_OPEN_HOLD_STEPS = 500

APPROACH_PREGRASP_STEPS = 2000

PREGRASP_HOLD_STEPS = 500

APPROACH_GRASP_STEPS = 1500

PRE_CLOSE_HOLD_STEPS = 1000

GRIPPER_CLOSE_STEPS = 2000

POST_GRASP_HOLD_STEPS = 1000

LIFT_STEPS = 2500

PRE_TRANSPORT_HOLD_STEPS = 1000

POST_TRANSPORT_HOLD_STEPS = 1500

POST_PLACEMENT_HOLD_STEPS = 2000

RELEASE_STEPS = 3000

POST_RELEASE_HOLD_STEPS = 2000

POST_RETREAT_HOLD_STEPS = 1500


# ============================================================
# IK acceptance criteria
# ============================================================

IK_POSITION_TOLERANCE_M = 1.0e-3

IK_ORIENTATION_TOLERANCE_DEG = 1.0


# ============================================================
# Gripper commands
# ============================================================

OPEN_GRIPPER_CTRL = 255.0

CLOSE_GRIPPER_CTRL = 0.0


# ============================================================
# Goal geometry
# ============================================================

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


# ============================================================
# Acceptance thresholds
# ============================================================

TRANSPORT_X_TOLERANCE_M = 0.010

TRANSPORT_Y_TOLERANCE_M = 0.010

TRANSPORT_Z_TOLERANCE_M = 0.010

MAX_TRANSPORT_ORIENTATION_CHANGE_DEG = 2.0

MAX_TRANSPORT_LINEAR_VELOCITY_MPS = 0.01

MAX_TRANSPORT_ANGULAR_VELOCITY_RADPS = 0.10


MIN_PLACEMENT_TABLE_SUPPORT_N = 1.0

MAX_PLACEMENT_LINEAR_VELOCITY_MPS = 0.01

MAX_PLACEMENT_ANGULAR_VELOCITY_RADPS = 0.10

MAX_PLACEMENT_ORIENTATION_CHANGE_DEG = 2.0


MIN_RELEASE_TABLE_SUPPORT_N = 18.0

MAX_RELEASE_FINGER_VERTICAL_SUPPORT_N = 1.0

MAX_RELEASE_FINGER_NORMAL_FORCE_N = 0.5

MAX_RELEASE_LINEAR_VELOCITY_MPS = 0.01

MAX_RELEASE_ANGULAR_VELOCITY_RADPS = 0.10

MAX_RELEASE_ROLL_PITCH_DEG = 2.0


MIN_FINAL_TABLE_SUPPORT_N = 18.0

MAX_FINAL_FINGER_NORMAL_FORCE_N = 0.5

MAX_FINAL_LINEAR_VELOCITY_MPS = 0.01

MAX_FINAL_ANGULAR_VELOCITY_RADPS = 0.10

MAX_FINAL_ROLL_PITCH_DEG = 2.0

MAX_RETREAT_OBJECT_DISPLACEMENT_M = 0.005


# ============================================================
# Model names
# ============================================================

LEFT_JOINT_NAMES = [
    f"left_joint{i}"
    for i in range(1, 8)
]

RIGHT_JOINT_NAMES = [
    f"right_joint{i}"
    for i in range(1, 8)
]


LEFT_EE_SITE_NAME = "left_ee_site"

RIGHT_EE_SITE_NAME = "right_ee_site"

OBJECT_BODY_NAME = "transport_object"

OBJECT_FREEJOINT_NAME = "object_freejoint"

OBJECT_CENTER_SITE_NAME = "object_center"

TABLE_GEOM_NAME = "work_table"


# ============================================================
# Generic helpers
# ============================================================

def rotation_matrix_to_rpy(
    rotation,
):
    """
    Convert a 3x3 rotation matrix to roll, pitch and yaw.
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


def object_state(
    model,
    data,
):
    """
    Return object position, orientation and velocities.
    """

    object_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        OBJECT_BODY_NAME,
    )

    if object_body_id < 0:
        raise RuntimeError(
            "Transport object body was not found."
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
        OBJECT_FREEJOINT_NAME,
    )

    if object_joint_id < 0:
        raise RuntimeError(
            "Transport object free joint was not found."
        )


    dof_address = int(
        model.jnt_dofadr[
            object_joint_id
        ]
    )


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


def get_object_support_forces(
    model,
    data,
):
    """
    Return contact support acting on the transport object.

    Returns
    -------
    finger_fz
        Total vertical force on the object from fingers.

    table_fz
        Total vertical force on the object from the table.

    finger_normal_sum
        Sum of finger-object contact normal magnitudes.

    finger_contact_count
        Number of active finger-object contacts.
    """

    finger_fz = 0.0

    table_fz = 0.0

    finger_normal_sum = 0.0

    finger_contact_count = 0


    for contact_id in range(
        data.ncon
    ):

        contact = data.contact[
            contact_id
        ]

        geom1_id = int(
            contact.geom1
        )

        geom2_id = int(
            contact.geom2
        )


        body1_id = int(
            model.geom_bodyid[
                geom1_id
            ]
        )

        body2_id = int(
            model.geom_bodyid[
                geom2_id
            ]
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


        if (
            body1_name
            != OBJECT_BODY_NAME
            and body2_name
            != OBJECT_BODY_NAME
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
                :3
            ].copy()
        )


        contact_frame = (
            contact.frame
            .reshape(
                3,
                3,
            )
        )


        force_world_on_geom2 = (
            contact_frame.T
            @ contact_force_local
        )

        force_world_on_geom1 = (
            -force_world_on_geom2
        )


        if (
            body1_name
            == OBJECT_BODY_NAME
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


        is_finger_contact = (
            other_body_name is not None
            and "finger"
            in other_body_name.lower()
        )


        is_table_contact = (
            other_body_name == "world"
            and (
                geom1_name
                == TABLE_GEOM_NAME
                or geom2_name
                == TABLE_GEOM_NAME
            )
        )


        if is_finger_contact:

            finger_contact_count += 1

            finger_fz += float(
                object_force_world[2]
            )

            finger_normal_sum += abs(
                float(
                    contact_force_local[0]
                )
            )


        if is_table_contact:

            table_fz += float(
                object_force_world[2]
            )


    return (
        finger_fz,
        table_fz,
        finger_normal_sum,
        finger_contact_count,
    )


def goal_containment(
    object_position,
):
    """
    Check full-footprint containment of the approximately
    flat object inside the benchmark goal region.
    """

    goal_x_pass = (
        abs(
            object_position[0]
            - GOAL_CENTER_X
        )
        <= GOAL_X_CENTER_TOLERANCE
    )


    goal_y_pass = (
        abs(
            object_position[1]
            - GOAL_CENTER_Y
        )
        <= GOAL_Y_CENTER_TOLERANCE
    )


    return (
        goal_x_pass,
        goal_y_pass,
    )


def validate_ik_solution(
    target_position,
    target_rotation,
    achieved_position,
    achieved_rotation,
):
    """
    Validate one solved end-effector pose.
    """

    position_error = float(
        np.linalg.norm(
            achieved_position
            - target_position
        )
    )


    orientation_error = (
        ClassicalController.orientation_error_deg(
            target_rotation,
            achieved_rotation,
        )
    )


    valid = all([
        position_error
        <= IK_POSITION_TOLERANCE_M,

        orientation_error
        <= IK_ORIENTATION_TOLERANCE_DEG,
    ])


    return (
        valid,
        position_error,
        orientation_error,
    )


def print_phase_result(
    phase_name,
    passed,
):
    """
    Print one compact baseline phase result.
    """

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{phase_name:<12} {status}"
    )
    
# ============================================================
# IK / trajectory generation
# ============================================================

def solve_checked_pose(
    model,
    data,
    joint_names,
    site_name,
    target_position,
    target_rotation,
    initial_q,
    label,
):
    """
    Solve one pose-IK target and immediately verify that
    the achieved pose satisfies the baseline tolerances.
    """

    (
        solved_q,
        achieved_position,
        achieved_rotation,
    ) = solve_pose_ik(
        model=model,
        data=data,
        joint_names=joint_names,
        site_name=site_name,
        target_position=target_position,
        target_rotation=target_rotation,
        initial_q=initial_q,
    )


    (
        valid,
        position_error,
        orientation_error,
    ) = validate_ik_solution(
        target_position,
        target_rotation,
        achieved_position,
        achieved_rotation,
    )


    if not valid:
        raise RuntimeError(
            f"{label} IK validation failed: "
            f"position error = "
            f"{position_error * 1000.0:.3f} mm, "
            f"orientation error = "
            f"{orientation_error:.4f} deg."
        )


    return solved_q.copy()


def build_task_trajectories(
    model,
    data,
    controller,
):
    """
    Build every IK configuration needed by the validated
    classical cooperative manipulation sequence.

    The nominal end-effector orientations are taken from the
    initial dual-Panda configuration and preserved through the
    complete task.

    Returns
    -------
    dict
        All solved task configurations and joint trajectories.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BUILDING CLASSICAL BASELINE TRAJECTORIES"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Establish nominal initial state
    # --------------------------------------------------------

    controller.set_arm_qpos(
        INITIAL_ARM_Q,
        INITIAL_ARM_Q,
    )

    controller.open_grippers()


    mujoco.mj_forward(
        model,
        data,
    )


    left_ee_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        LEFT_EE_SITE_NAME,
    )

    right_ee_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        RIGHT_EE_SITE_NAME,
    )


    if left_ee_site_id < 0:
        raise RuntimeError(
            "Left end-effector site was not found."
        )

    if right_ee_site_id < 0:
        raise RuntimeError(
            "Right end-effector site was not found."
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


    # --------------------------------------------------------
    # Pregrasp
    # --------------------------------------------------------

    print(
        "\nSolving pregrasp IK..."
    )


    left_pregrasp_q = solve_checked_pose(
        model=model,
        data=data,
        joint_names=LEFT_JOINT_NAMES,
        site_name=LEFT_EE_SITE_NAME,
        target_position=LEFT_PREGRASP_POSITION,
        target_rotation=left_target_rotation,
        initial_q=INITIAL_ARM_Q,
        label="Left pregrasp",
    )


    right_pregrasp_q = solve_checked_pose(
        model=model,
        data=data,
        joint_names=RIGHT_JOINT_NAMES,
        site_name=RIGHT_EE_SITE_NAME,
        target_position=RIGHT_PREGRASP_POSITION,
        target_rotation=right_target_rotation,
        initial_q=INITIAL_ARM_Q,
        label="Right pregrasp",
    )


    print(
        "Pregrasp IK: PASS"
    )


    # --------------------------------------------------------
    # Grasp
    # --------------------------------------------------------

    print(
        "Solving grasp IK..."
    )


    left_grasp_q = solve_checked_pose(
        model=model,
        data=data,
        joint_names=LEFT_JOINT_NAMES,
        site_name=LEFT_EE_SITE_NAME,
        target_position=LEFT_GRASP_POSITION,
        target_rotation=left_target_rotation,
        initial_q=left_pregrasp_q,
        label="Left grasp",
    )


    right_grasp_q = solve_checked_pose(
        model=model,
        data=data,
        joint_names=RIGHT_JOINT_NAMES,
        site_name=RIGHT_EE_SITE_NAME,
        target_position=RIGHT_GRASP_POSITION,
        target_rotation=right_target_rotation,
        initial_q=right_pregrasp_q,
        label="Right grasp",
    )


    print(
        "Grasp IK: PASS"
    )


    # --------------------------------------------------------
    # Lift
    # --------------------------------------------------------

    print(
        "Solving lift IK..."
    )


    left_lift_q = solve_checked_pose(
        model=model,
        data=data,
        joint_names=LEFT_JOINT_NAMES,
        site_name=LEFT_EE_SITE_NAME,
        target_position=LEFT_LIFT_POSITION,
        target_rotation=left_target_rotation,
        initial_q=left_grasp_q,
        label="Left lift",
    )


    right_lift_q = solve_checked_pose(
        model=model,
        data=data,
        joint_names=RIGHT_JOINT_NAMES,
        site_name=RIGHT_EE_SITE_NAME,
        target_position=RIGHT_LIFT_POSITION,
        target_rotation=right_target_rotation,
        initial_q=right_grasp_q,
        label="Right lift",
    )


    print(
        "Lift IK: PASS"
    )


    # ========================================================
    # Flat horizontal transport trajectory
    # ========================================================

    print(
        "\nBuilding horizontal transport trajectory..."
    )


    left_transport_waypoints = [
        left_lift_q.copy()
    ]

    right_transport_waypoints = [
        right_lift_q.copy()
    ]


    previous_left_q = (
        left_lift_q.copy()
    )

    previous_right_q = (
        right_lift_q.copy()
    )


    max_transport_position_error = 0.0

    max_transport_orientation_error = 0.0


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


        trajectory_alpha = (
            ClassicalController.smoothstep(
                raw_alpha
            )
        )


        current_x = (
            TRANSPORT_X
            * trajectory_alpha
        )


        left_target_position = np.array([
            current_x,
            LEFT_LIFT_POSITION[1],
            TRANSPORT_Z,
        ])


        right_target_position = np.array([
            current_x,
            RIGHT_LIFT_POSITION[1],
            TRANSPORT_Z,
        ])


        (
            left_waypoint_q,
            left_achieved_position,
            left_achieved_rotation,
        ) = solve_pose_ik(
            model=model,
            data=data,
            joint_names=LEFT_JOINT_NAMES,
            site_name=LEFT_EE_SITE_NAME,
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
            joint_names=RIGHT_JOINT_NAMES,
            site_name=RIGHT_EE_SITE_NAME,
            target_position=right_target_position,
            target_rotation=right_target_rotation,
            initial_q=previous_right_q,
        )


        (
            left_valid,
            left_position_error,
            left_orientation_error,
        ) = validate_ik_solution(
            left_target_position,
            left_target_rotation,
            left_achieved_position,
            left_achieved_rotation,
        )


        (
            right_valid,
            right_position_error,
            right_orientation_error,
        ) = validate_ik_solution(
            right_target_position,
            right_target_rotation,
            right_achieved_position,
            right_achieved_rotation,
        )


        waypoint_position_error = max(
            left_position_error,
            right_position_error,
        )


        waypoint_orientation_error = max(
            left_orientation_error,
            right_orientation_error,
        )


        max_transport_position_error = max(
            max_transport_position_error,
            waypoint_position_error,
        )


        max_transport_orientation_error = max(
            max_transport_orientation_error,
            waypoint_orientation_error,
        )


        if not (
            left_valid
            and right_valid
        ):
            raise RuntimeError(
                "Transport trajectory IK failed at "
                f"waypoint {waypoint_index}: "
                f"position error = "
                f"{waypoint_position_error * 1000.0:.3f} mm, "
                f"orientation error = "
                f"{waypoint_orientation_error:.4f} deg."
            )


        left_transport_waypoints.append(
            left_waypoint_q.copy()
        )

        right_transport_waypoints.append(
            right_waypoint_q.copy()
        )


        previous_left_q = (
            left_waypoint_q.copy()
        )

        previous_right_q = (
            right_waypoint_q.copy()
        )


    if (
        len(
            left_transport_waypoints
        )
        != TRANSPORT_TRAJECTORY_WAYPOINTS
        or len(
            right_transport_waypoints
        )
        != TRANSPORT_TRAJECTORY_WAYPOINTS
    ):
        raise RuntimeError(
            "Unexpected number of transport waypoints."
        )


    left_transport_q = (
        left_transport_waypoints[
            -1
        ].copy()
    )

    right_transport_q = (
        right_transport_waypoints[
            -1
        ].copy()
    )


    print(
        "Horizontal transport trajectory: PASS"
    )

    print(
        "  waypoints:",
        TRANSPORT_TRAJECTORY_WAYPOINTS,
    )

    print(
        "  max IK position error [mm]:",
        max_transport_position_error
        * 1000.0,
    )

    print(
        "  max IK orientation error [deg]:",
        max_transport_orientation_error,
    )


    # ========================================================
    # Diagonal flat-placement trajectory
    # ========================================================

    print(
        "\nBuilding flat-placement trajectory..."
    )


    left_placement_waypoints = [
        left_transport_q.copy()
    ]

    right_placement_waypoints = [
        right_transport_q.copy()
    ]


    previous_left_q = (
        left_transport_q.copy()
    )

    previous_right_q = (
        right_transport_q.copy()
    )


    max_placement_position_error = 0.0

    max_placement_orientation_error = 0.0


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


        trajectory_alpha = (
            ClassicalController.smoothstep(
                raw_alpha
            )
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
            LEFT_PLACEMENT_POSITION[1],
            current_z,
        ])


        right_target_position = np.array([
            current_x,
            RIGHT_PLACEMENT_POSITION[1],
            current_z,
        ])


        (
            left_waypoint_q,
            left_achieved_position,
            left_achieved_rotation,
        ) = solve_pose_ik(
            model=model,
            data=data,
            joint_names=LEFT_JOINT_NAMES,
            site_name=LEFT_EE_SITE_NAME,
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
            joint_names=RIGHT_JOINT_NAMES,
            site_name=RIGHT_EE_SITE_NAME,
            target_position=right_target_position,
            target_rotation=right_target_rotation,
            initial_q=previous_right_q,
        )


        (
            left_valid,
            left_position_error,
            left_orientation_error,
        ) = validate_ik_solution(
            left_target_position,
            left_target_rotation,
            left_achieved_position,
            left_achieved_rotation,
        )


        (
            right_valid,
            right_position_error,
            right_orientation_error,
        ) = validate_ik_solution(
            right_target_position,
            right_target_rotation,
            right_achieved_position,
            right_achieved_rotation,
        )


        waypoint_position_error = max(
            left_position_error,
            right_position_error,
        )


        waypoint_orientation_error = max(
            left_orientation_error,
            right_orientation_error,
        )


        max_placement_position_error = max(
            max_placement_position_error,
            waypoint_position_error,
        )


        max_placement_orientation_error = max(
            max_placement_orientation_error,
            waypoint_orientation_error,
        )


        if not (
            left_valid
            and right_valid
        ):
            raise RuntimeError(
                "Placement trajectory IK failed at "
                f"waypoint {waypoint_index}: "
                f"position error = "
                f"{waypoint_position_error * 1000.0:.3f} mm, "
                f"orientation error = "
                f"{waypoint_orientation_error:.4f} deg."
            )


        left_placement_waypoints.append(
            left_waypoint_q.copy()
        )

        right_placement_waypoints.append(
            right_waypoint_q.copy()
        )


        previous_left_q = (
            left_waypoint_q.copy()
        )

        previous_right_q = (
            right_waypoint_q.copy()
        )


    if (
        len(
            left_placement_waypoints
        )
        != PLACEMENT_TRAJECTORY_WAYPOINTS
        or len(
            right_placement_waypoints
        )
        != PLACEMENT_TRAJECTORY_WAYPOINTS
    ):
        raise RuntimeError(
            "Unexpected number of placement waypoints."
        )


    left_placement_q = (
        left_placement_waypoints[
            -1
        ].copy()
    )

    right_placement_q = (
        right_placement_waypoints[
            -1
        ].copy()
    )


    print(
        "Flat-placement trajectory: PASS"
    )

    print(
        "  waypoints:",
        PLACEMENT_TRAJECTORY_WAYPOINTS,
    )

    print(
        "  max IK position error [mm]:",
        max_placement_position_error
        * 1000.0,
    )

    print(
        "  max IK orientation error [deg]:",
        max_placement_orientation_error,
    )


    # ========================================================
    # Outward arm-retreat trajectory
    # ========================================================

    print(
        "\nBuilding outward retreat trajectory..."
    )


    left_retreat_waypoints = [
        left_placement_q.copy()
    ]

    right_retreat_waypoints = [
        right_placement_q.copy()
    ]


    previous_left_q = (
        left_placement_q.copy()
    )

    previous_right_q = (
        right_placement_q.copy()
    )


    max_retreat_position_error = 0.0

    max_retreat_orientation_error = 0.0


    for waypoint_index in range(
        1,
        RETREAT_TRAJECTORY_WAYPOINTS,
    ):

        raw_alpha = (
            waypoint_index
            / (
                RETREAT_TRAJECTORY_WAYPOINTS
                - 1
            )
        )


        trajectory_alpha = (
            ClassicalController.smoothstep(
                raw_alpha
            )
        )


        current_left_y = (
            LEFT_PLACEMENT_POSITION[1]
            + (
                LEFT_RETREAT_POSITION[1]
                - LEFT_PLACEMENT_POSITION[1]
            )
            * trajectory_alpha
        )


        current_right_y = (
            RIGHT_PLACEMENT_POSITION[1]
            + (
                RIGHT_RETREAT_POSITION[1]
                - RIGHT_PLACEMENT_POSITION[1]
            )
            * trajectory_alpha
        )


        left_target_position = np.array([
            PLACEMENT_X,
            current_left_y,
            PLACEMENT_Z,
        ])


        right_target_position = np.array([
            PLACEMENT_X,
            current_right_y,
            PLACEMENT_Z,
        ])


        (
            left_waypoint_q,
            left_achieved_position,
            left_achieved_rotation,
        ) = solve_pose_ik(
            model=model,
            data=data,
            joint_names=LEFT_JOINT_NAMES,
            site_name=LEFT_EE_SITE_NAME,
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
            joint_names=RIGHT_JOINT_NAMES,
            site_name=RIGHT_EE_SITE_NAME,
            target_position=right_target_position,
            target_rotation=right_target_rotation,
            initial_q=previous_right_q,
        )


        (
            left_valid,
            left_position_error,
            left_orientation_error,
        ) = validate_ik_solution(
            left_target_position,
            left_target_rotation,
            left_achieved_position,
            left_achieved_rotation,
        )


        (
            right_valid,
            right_position_error,
            right_orientation_error,
        ) = validate_ik_solution(
            right_target_position,
            right_target_rotation,
            right_achieved_position,
            right_achieved_rotation,
        )


        waypoint_position_error = max(
            left_position_error,
            right_position_error,
        )


        waypoint_orientation_error = max(
            left_orientation_error,
            right_orientation_error,
        )


        max_retreat_position_error = max(
            max_retreat_position_error,
            waypoint_position_error,
        )


        max_retreat_orientation_error = max(
            max_retreat_orientation_error,
            waypoint_orientation_error,
        )


        if not (
            left_valid
            and right_valid
        ):
            raise RuntimeError(
                "Retreat trajectory IK failed at "
                f"waypoint {waypoint_index}: "
                f"position error = "
                f"{waypoint_position_error * 1000.0:.3f} mm, "
                f"orientation error = "
                f"{waypoint_orientation_error:.4f} deg."
            )


        left_retreat_waypoints.append(
            left_waypoint_q.copy()
        )

        right_retreat_waypoints.append(
            right_waypoint_q.copy()
        )


        previous_left_q = (
            left_waypoint_q.copy()
        )

        previous_right_q = (
            right_waypoint_q.copy()
        )


    if (
        len(
            left_retreat_waypoints
        )
        != RETREAT_TRAJECTORY_WAYPOINTS
        or len(
            right_retreat_waypoints
        )
        != RETREAT_TRAJECTORY_WAYPOINTS
    ):
        raise RuntimeError(
            "Unexpected number of retreat waypoints."
        )


    left_retreat_q = (
        left_retreat_waypoints[
            -1
        ].copy()
    )

    right_retreat_q = (
        right_retreat_waypoints[
            -1
        ].copy()
    )


    print(
        "Outward retreat trajectory: PASS"
    )

    print(
        "  waypoints:",
        RETREAT_TRAJECTORY_WAYPOINTS,
    )

    print(
        "  max IK position error [mm]:",
        max_retreat_position_error
        * 1000.0,
    )

    print(
        "  max IK orientation error [deg]:",
        max_retreat_orientation_error,
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "ALL CLASSICAL BASELINE IK TRAJECTORIES: PASS"
    )

    print(
        "=" * 70
    )


    return {
        "left_target_rotation":
            left_target_rotation,

        "right_target_rotation":
            right_target_rotation,

        "left_pregrasp_q":
            left_pregrasp_q,

        "right_pregrasp_q":
            right_pregrasp_q,

        "left_grasp_q":
            left_grasp_q,

        "right_grasp_q":
            right_grasp_q,

        "left_lift_q":
            left_lift_q,

        "right_lift_q":
            right_lift_q,

        "left_transport_waypoints":
            left_transport_waypoints,

        "right_transport_waypoints":
            right_transport_waypoints,

        "left_transport_q":
            left_transport_q,

        "right_transport_q":
            right_transport_q,

        "left_placement_waypoints":
            left_placement_waypoints,

        "right_placement_waypoints":
            right_placement_waypoints,

        "left_placement_q":
            left_placement_q,

        "right_placement_q":
            right_placement_q,

        "left_retreat_waypoints":
            left_retreat_waypoints,

        "right_retreat_waypoints":
            right_retreat_waypoints,

        "left_retreat_q":
            left_retreat_q,

        "right_retreat_q":
            right_retreat_q,
    }
    
# ============================================================
# Classical baseline execution
# ============================================================

def run_classical_baseline():
    """
    Execute the complete validated classical cooperative
    dual-arm transport baseline.

    Sequence
    --------
    APPROACH
        Initial -> pregrasp -> grasp pose.

    GRASP
        Close both grippers and establish bilateral support.

    LIFT
        Raise the object by the validated 2 cm lift.

    TRANSPORT
        Move the suspended object horizontally to the
        validated transport configuration.

    PLACE
        Move diagonally forward/downward to the flat
        placement pose and establish table contact.

    RELEASE
        Gradually open both grippers while holding the
        placement pose.

    RETREAT
        Move the open grippers laterally away from the
        released object.

    SUCCESS
        Verify the final goal, support, contact, velocity,
        orientation and retreat-displacement criteria.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CLASSICAL COOPERATIVE TRANSPORT BASELINE"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(
            MODEL_PATH
        )
    )

    data = mujoco.MjData(
        model
    )


    controller = ClassicalController(
        model,
        data,
    )


    print(
        "\nModel:"
    )

    print(
        "  nq:",
        model.nq,
    )

    print(
        "  nv:",
        model.nv,
    )

    print(
        "  nu:",
        model.nu,
    )

    print(
        "  timestep [s]:",
        model.opt.timestep,
    )


    # --------------------------------------------------------
    # Build and validate all IK trajectories before executing
    # any dynamics.
    # --------------------------------------------------------

    trajectories = (
        build_task_trajectories(
            model,
            data,
            controller,
        )
    )


    # --------------------------------------------------------
    # Reset after IK.
    #
    # solve_pose_ik modifies data.qpos during optimization,
    # therefore the actual dynamic experiment must always
    # begin from a clean MuJoCo reset.
    # --------------------------------------------------------

    mujoco.mj_resetData(
        model,
        data,
    )


    controller.set_arm_qpos(
        INITIAL_ARM_Q,
        INITIAL_ARM_Q,
    )

    controller.open_grippers()


    mujoco.mj_forward(
        model,
        data,
    )


    # ========================================================
    # Global benchmark bookkeeping
    # ========================================================

    baseline_wall_start = (
        time.perf_counter()
    )


    phase_results = {}


    # Keep a few generic metrics that are immediately useful
    # for later comparison with learned policies.
    #
    # These are descriptive metrics. They do NOT alter any
    # acceptance threshold or controller behaviour.

    maximum_roll_pitch_deg = 0.0

    maximum_object_linear_speed = 0.0

    maximum_object_angular_speed = 0.0

    control_effort_integral = 0.0


    previous_arm_ctrl = None

    command_variation_integral = 0.0


    all_arm_actuator_ids = (
        controller.left_arm_actuator_ids
        + controller.right_arm_actuator_ids
    )


    def observe_global_state(
        _step_index=None,
    ):
        """
        Record task-wide benchmark quantities after a physics
        step without modifying the simulation state.
        """

        nonlocal maximum_roll_pitch_deg

        nonlocal maximum_object_linear_speed

        nonlocal maximum_object_angular_speed

        nonlocal control_effort_integral

        nonlocal previous_arm_ctrl

        nonlocal command_variation_integral


        (
            _position,
            rotation,
            linear_velocity,
            angular_velocity,
        ) = object_state(
            model,
            data,
        )


        rpy = rotation_matrix_to_rpy(
            rotation
        )


        roll_pitch_deg = max(
            abs(
                float(
                    np.degrees(
                        rpy[0]
                    )
                )
            ),
            abs(
                float(
                    np.degrees(
                        rpy[1]
                    )
                )
            ),
        )


        maximum_roll_pitch_deg = max(
            maximum_roll_pitch_deg,
            roll_pitch_deg,
        )


        maximum_object_linear_speed = max(
            maximum_object_linear_speed,
            float(
                np.linalg.norm(
                    linear_velocity
                )
            ),
        )


        maximum_object_angular_speed = max(
            maximum_object_angular_speed,
            float(
                np.linalg.norm(
                    angular_velocity
                )
            ),
        )


        arm_force = np.asarray(
            data.actuator_force[
                all_arm_actuator_ids
            ],
            dtype=float,
        )


        control_effort_integral += (
            float(
                np.sum(
                    np.square(
                        arm_force
                    )
                )
            )
            * model.opt.timestep
        )


        current_arm_ctrl = np.asarray(
            data.ctrl[
                all_arm_actuator_ids
            ],
            dtype=float,
        ).copy()


        if previous_arm_ctrl is not None:

            command_difference = (
                current_arm_ctrl
                - previous_arm_ctrl
            )

            command_variation_integral += (
                float(
                    np.sum(
                        np.square(
                            command_difference
                        )
                    )
                )
            )


        previous_arm_ctrl = (
            current_arm_ctrl
        )


    # ========================================================
    # APPROACH
    # ========================================================

    controller.set_phase(
        ClassicalPhase.APPROACH
    )


    print(
        "\n[APPROACH]"
    )


    controller.hold_pose(
        INITIAL_ARM_Q,
        INITIAL_ARM_Q,
        INITIAL_OPEN_HOLD_STEPS,
        gripper_ctrl=OPEN_GRIPPER_CTRL,
        step_callback=observe_global_state,
    )


    controller.execute_joint_trajectory(
        INITIAL_ARM_Q,
        INITIAL_ARM_Q,
        trajectories[
            "left_pregrasp_q"
        ],
        trajectories[
            "right_pregrasp_q"
        ],
        APPROACH_PREGRASP_STEPS,
        gripper_ctrl=OPEN_GRIPPER_CTRL,
        step_callback=observe_global_state,
    )


    controller.hold_pose(
        trajectories[
            "left_pregrasp_q"
        ],
        trajectories[
            "right_pregrasp_q"
        ],
        PREGRASP_HOLD_STEPS,
        gripper_ctrl=OPEN_GRIPPER_CTRL,
        step_callback=observe_global_state,
    )


    controller.execute_joint_trajectory(
        trajectories[
            "left_pregrasp_q"
        ],
        trajectories[
            "right_pregrasp_q"
        ],
        trajectories[
            "left_grasp_q"
        ],
        trajectories[
            "right_grasp_q"
        ],
        APPROACH_GRASP_STEPS,
        gripper_ctrl=OPEN_GRIPPER_CTRL,
        step_callback=observe_global_state,
    )


    controller.hold_pose(
        trajectories[
            "left_grasp_q"
        ],
        trajectories[
            "right_grasp_q"
        ],
        PRE_CLOSE_HOLD_STEPS,
        gripper_ctrl=OPEN_GRIPPER_CTRL,
        step_callback=observe_global_state,
    )


    phase_results[
        "APPROACH"
    ] = True


    print_phase_result(
        "APPROACH",
        True,
    )


    # ========================================================
    # GRASP
    # ========================================================

    controller.set_phase(
        ClassicalPhase.GRASP
    )


    print(
        "\n[GRASP]"
    )


    controller.hold_pose(
        trajectories[
            "left_grasp_q"
        ],
        trajectories[
            "right_grasp_q"
        ],
        GRIPPER_CLOSE_STEPS,
        gripper_ctrl=CLOSE_GRIPPER_CTRL,
        step_callback=observe_global_state,
    )


    controller.hold_pose(
        trajectories[
            "left_grasp_q"
        ],
        trajectories[
            "right_grasp_q"
        ],
        POST_GRASP_HOLD_STEPS,
        gripper_ctrl=CLOSE_GRIPPER_CTRL,
        step_callback=observe_global_state,
    )


    mujoco.mj_forward(
        model,
        data,
    )


    (
        grasp_finger_fz,
        grasp_table_fz,
        grasp_finger_normal,
        grasp_finger_contacts,
    ) = get_object_support_forces(
        model,
        data,
    )


    grasp_pass = all([
        grasp_finger_contacts > 0,
        grasp_finger_normal > 0.0,
    ])


    phase_results[
        "GRASP"
    ] = grasp_pass


    print(
        "  finger contacts:",
        grasp_finger_contacts,
    )

    print(
        "  finger normal-force sum [N]:",
        grasp_finger_normal,
    )

    print_phase_result(
        "GRASP",
        grasp_pass,
    )


    if not grasp_pass:

        controller.set_phase(
            ClassicalPhase.FAILURE
        )

        raise RuntimeError(
            "Classical bilateral grasp failed."
        )


    # ========================================================
    # LIFT
    # ========================================================

    controller.set_phase(
        ClassicalPhase.LIFT
    )


    print(
        "\n[LIFT]"
    )


    (
        object_position_before_lift,
        _,
        _,
        _,
    ) = object_state(
        model,
        data,
    )


    controller.execute_joint_trajectory(
        trajectories[
            "left_grasp_q"
        ],
        trajectories[
            "right_grasp_q"
        ],
        trajectories[
            "left_lift_q"
        ],
        trajectories[
            "right_lift_q"
        ],
        LIFT_STEPS,
        gripper_ctrl=CLOSE_GRIPPER_CTRL,
        step_callback=observe_global_state,
    )


    controller.hold_pose(
        trajectories[
            "left_lift_q"
        ],
        trajectories[
            "right_lift_q"
        ],
        PRE_TRANSPORT_HOLD_STEPS,
        gripper_ctrl=CLOSE_GRIPPER_CTRL,
        step_callback=observe_global_state,
    )


    mujoco.mj_forward(
        model,
        data,
    )


    (
        object_position_after_lift,
        _,
        object_linear_velocity_after_lift,
        object_angular_velocity_after_lift,
    ) = object_state(
        model,
        data,
    )


    (
        lift_finger_fz,
        lift_table_fz,
        lift_finger_normal,
        lift_finger_contacts,
    ) = get_object_support_forces(
        model,
        data,
    )


    lift_delta_z = (
        object_position_after_lift[2]
        - object_position_before_lift[2]
    )


    lift_pass = all([
        lift_delta_z
        >= 0.010,

        lift_table_fz
        <= 0.10,

        lift_finger_contacts
        > 0,

        np.linalg.norm(
            object_linear_velocity_after_lift
        )
        <= 0.01,

        np.linalg.norm(
            object_angular_velocity_after_lift
        )
        <= 0.10,
    ])


    phase_results[
        "LIFT"
    ] = lift_pass


    print(
        "  object Z change [m]:",
        lift_delta_z,
    )

    print(
        "  table support [N]:",
        lift_table_fz,
    )

    print(
        "  finger support [N]:",
        lift_finger_fz,
    )

    print_phase_result(
        "LIFT",
        lift_pass,
    )


    if not lift_pass:

        controller.set_phase(
            ClassicalPhase.FAILURE
        )

        raise RuntimeError(
            "Classical cooperative lift failed."
        )


    # ========================================================
    # TRANSPORT
    # ========================================================

    controller.set_phase(
        ClassicalPhase.TRANSPORT
    )


    print(
        "\n[TRANSPORT]"
    )


    (
        object_position_before_transport,
        object_rotation_before_transport,
        _,
        _,
    ) = object_state(
        model,
        data,
    )


    transport_table_contact = False

    maximum_transport_orientation_change_deg = 0.0


    def observe_transport(
        step_index,
    ):
        nonlocal transport_table_contact

        nonlocal maximum_transport_orientation_change_deg


        observe_global_state(
            step_index
        )


        (
            _finger_fz,
            table_fz,
            _finger_normal,
            _finger_contacts,
        ) = get_object_support_forces(
            model,
            data,
        )


        if table_fz > 0.01:
            transport_table_contact = True


        (
            _position,
            rotation,
            _linear_velocity,
            _angular_velocity,
        ) = object_state(
            model,
            data,
        )


        orientation_change = (
            ClassicalController.orientation_error_deg(
                object_rotation_before_transport,
                rotation,
            )
        )


        maximum_transport_orientation_change_deg = max(
            maximum_transport_orientation_change_deg,
            orientation_change,
        )


    controller.execute_waypoint_trajectory(
        trajectories[
            "left_transport_waypoints"
        ],
        trajectories[
            "right_transport_waypoints"
        ],
        PHYSICS_STEPS_PER_WAYPOINT_SEGMENT,
        gripper_ctrl=CLOSE_GRIPPER_CTRL,
        step_callback=observe_transport,
        smooth_interpolation=False,
    )


    controller.hold_pose(
        trajectories[
            "left_transport_q"
        ],
        trajectories[
            "right_transport_q"
        ],
        POST_TRANSPORT_HOLD_STEPS,
        gripper_ctrl=CLOSE_GRIPPER_CTRL,
        step_callback=observe_transport,
    )


    mujoco.mj_forward(
        model,
        data,
    )


    (
        object_position_after_transport,
        object_rotation_after_transport,
        object_linear_velocity_after_transport,
        object_angular_velocity_after_transport,
    ) = object_state(
        model,
        data,
    )


    final_transport_orientation_change_deg = (
        ClassicalController.orientation_error_deg(
            object_rotation_before_transport,
            object_rotation_after_transport,
        )
    )


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


    transport_linear_velocity_pass = (
        np.linalg.norm(
            object_linear_velocity_after_transport
        )
        <= MAX_TRANSPORT_LINEAR_VELOCITY_MPS
    )


    transport_angular_velocity_pass = (
        np.linalg.norm(
            object_angular_velocity_after_transport
        )
        <= MAX_TRANSPORT_ANGULAR_VELOCITY_RADPS
    )


    transport_orientation_pass = (
        final_transport_orientation_change_deg
        <= MAX_TRANSPORT_ORIENTATION_CHANGE_DEG
    )


    transport_pass = all([
        transport_x_pass,
        transport_y_pass,
        transport_z_pass,
        not transport_table_contact,
        transport_linear_velocity_pass,
        transport_angular_velocity_pass,
        transport_orientation_pass,
    ])


    phase_results[
        "TRANSPORT"
    ] = transport_pass


    print(
        "  final object position:",
        object_position_after_transport,
    )

    print(
        "  final orientation change [deg]:",
        final_transport_orientation_change_deg,
    )

    print(
        "  max orientation change [deg]:",
        maximum_transport_orientation_change_deg,
    )

    print(
        "  table contact occurred:",
        transport_table_contact,
    )

    print_phase_result(
        "TRANSPORT",
        transport_pass,
    )


    if not transport_pass:

        controller.set_phase(
            ClassicalPhase.FAILURE
        )

        raise RuntimeError(
            "Classical horizontal transport failed."
        )


    # ========================================================
    # PLACE
    # ========================================================

    controller.set_phase(
        ClassicalPhase.PLACE
    )


    print(
        "\n[PLACE]"
    )


    (
        object_position_before_placement,
        object_rotation_before_placement,
        _,
        _,
    ) = object_state(
        model,
        data,
    )


    placement_table_contact = False


    def observe_placement(
        step_index,
    ):
        nonlocal placement_table_contact


        observe_global_state(
            step_index
        )


        (
            _finger_fz,
            table_fz,
            _finger_normal,
            _finger_contacts,
        ) = get_object_support_forces(
            model,
            data,
        )


        if table_fz > 0.01:
            placement_table_contact = True


    controller.execute_waypoint_trajectory(
        trajectories[
            "left_placement_waypoints"
        ],
        trajectories[
            "right_placement_waypoints"
        ],
        PHYSICS_STEPS_PER_WAYPOINT_SEGMENT,
        gripper_ctrl=CLOSE_GRIPPER_CTRL,
        step_callback=observe_placement,
        smooth_interpolation=False,
    )


    controller.hold_pose(
        trajectories[
            "left_placement_q"
        ],
        trajectories[
            "right_placement_q"
        ],
        POST_PLACEMENT_HOLD_STEPS,
        gripper_ctrl=CLOSE_GRIPPER_CTRL,
        step_callback=observe_placement,
    )


    mujoco.mj_forward(
        model,
        data,
    )


    (
        object_position_after_placement,
        object_rotation_after_placement,
        object_linear_velocity_after_placement,
        object_angular_velocity_after_placement,
    ) = object_state(
        model,
        data,
    )


    (
        placement_finger_fz,
        placement_table_fz,
        placement_finger_normal,
        placement_finger_contacts,
    ) = get_object_support_forces(
        model,
        data,
    )


    (
        placement_goal_x_pass,
        placement_goal_y_pass,
    ) = goal_containment(
        object_position_after_placement
    )


    placement_orientation_change_deg = (
        ClassicalController.orientation_error_deg(
            object_rotation_before_placement,
            object_rotation_after_placement,
        )
    )


    placement_pass = all([
        placement_goal_x_pass,
        placement_goal_y_pass,
        placement_table_contact,
        placement_table_fz
        > MIN_PLACEMENT_TABLE_SUPPORT_N,
        np.linalg.norm(
            object_linear_velocity_after_placement
        )
        <= MAX_PLACEMENT_LINEAR_VELOCITY_MPS,
        np.linalg.norm(
            object_angular_velocity_after_placement
        )
        <= MAX_PLACEMENT_ANGULAR_VELOCITY_RADPS,
        placement_orientation_change_deg
        <= MAX_PLACEMENT_ORIENTATION_CHANGE_DEG,
    ])


    phase_results[
        "PLACE"
    ] = placement_pass


    print(
        "  object position:",
        object_position_after_placement,
    )

    print(
        "  table support [N]:",
        placement_table_fz,
    )

    print(
        "  finger support [N]:",
        placement_finger_fz,
    )

    print(
        "  inside goal X:",
        placement_goal_x_pass,
    )

    print(
        "  inside goal Y:",
        placement_goal_y_pass,
    )

    print_phase_result(
        "PLACE",
        placement_pass,
    )


    if not placement_pass:

        controller.set_phase(
            ClassicalPhase.FAILURE
        )

        raise RuntimeError(
            "Classical flat placement failed. "
            "Release was not started."
        )


    # ========================================================
    # RELEASE
    # ========================================================

    controller.set_phase(
        ClassicalPhase.RELEASE
    )


    print(
        "\n[RELEASE]"
    )


    (
        object_position_before_release,
        _,
        _,
        _,
    ) = object_state(
        model,
        data,
    )


    first_full_table_support_step = None

    first_zero_finger_contact_step = None


    release_observation_step = 0


    def observe_release(
        _step_index,
    ):
        nonlocal first_full_table_support_step

        nonlocal first_zero_finger_contact_step

        nonlocal release_observation_step


        release_observation_step += 1


        observe_global_state(
            release_observation_step
        )


        (
            _finger_fz,
            table_fz,
            _finger_normal,
            finger_contacts,
        ) = get_object_support_forces(
            model,
            data,
        )


        if (
            first_full_table_support_step
            is None
            and table_fz
            >= MIN_RELEASE_TABLE_SUPPORT_N
        ):
            first_full_table_support_step = (
                release_observation_step
            )


        if (
            first_zero_finger_contact_step
            is None
            and finger_contacts == 0
        ):
            first_zero_finger_contact_step = (
                release_observation_step
            )


    controller.execute_gripper_transition(
        trajectories[
            "left_placement_q"
        ],
        trajectories[
            "right_placement_q"
        ],
        CLOSE_GRIPPER_CTRL,
        OPEN_GRIPPER_CTRL,
        RELEASE_STEPS,
        step_callback=observe_release,
    )


    controller.hold_pose(
        trajectories[
            "left_placement_q"
        ],
        trajectories[
            "right_placement_q"
        ],
        POST_RELEASE_HOLD_STEPS,
        gripper_ctrl=OPEN_GRIPPER_CTRL,
        step_callback=observe_release,
    )


    mujoco.mj_forward(
        model,
        data,
    )


    (
        object_position_after_release,
        object_rotation_after_release,
        object_linear_velocity_after_release,
        object_angular_velocity_after_release,
    ) = object_state(
        model,
        data,
    )


    object_rpy_after_release = (
        rotation_matrix_to_rpy(
            object_rotation_after_release
        )
    )


    (
        released_finger_fz,
        released_table_fz,
        released_finger_normal,
        released_finger_contacts,
    ) = get_object_support_forces(
        model,
        data,
    )


    (
        release_goal_x_pass,
        release_goal_y_pass,
    ) = goal_containment(
        object_position_after_release
    )


    release_pass = all([
        release_goal_x_pass,
        release_goal_y_pass,

        released_table_fz
        >= MIN_RELEASE_TABLE_SUPPORT_N,

        abs(
            released_finger_fz
        )
        <= MAX_RELEASE_FINGER_VERTICAL_SUPPORT_N,

        released_finger_normal
        <= MAX_RELEASE_FINGER_NORMAL_FORCE_N,

        np.linalg.norm(
            object_linear_velocity_after_release
        )
        <= MAX_RELEASE_LINEAR_VELOCITY_MPS,

        np.linalg.norm(
            object_angular_velocity_after_release
        )
        <= MAX_RELEASE_ANGULAR_VELOCITY_RADPS,

        abs(
            np.degrees(
                object_rpy_after_release[0]
            )
        )
        <= MAX_RELEASE_ROLL_PITCH_DEG,

        abs(
            np.degrees(
                object_rpy_after_release[1]
            )
        )
        <= MAX_RELEASE_ROLL_PITCH_DEG,
    ])


    phase_results[
        "RELEASE"
    ] = release_pass


    print(
        "  position before:",
        object_position_before_release,
    )

    print(
        "  position after:",
        object_position_after_release,
    )

    print(
        "  displacement [m]:",
        object_position_after_release
        - object_position_before_release,
    )

    print(
        "  table support [N]:",
        released_table_fz,
    )

    print(
        "  finger vertical support [N]:",
        released_finger_fz,
    )

    print(
        "  finger normal-force sum [N]:",
        released_finger_normal,
    )

    print(
        "  finger contacts:",
        released_finger_contacts,
    )

    print(
        "  first full table-support step:",
        first_full_table_support_step,
    )

    print(
        "  first zero-finger-contact step:",
        first_zero_finger_contact_step,
    )

    print_phase_result(
        "RELEASE",
        release_pass,
    )


    if not release_pass:

        controller.set_phase(
            ClassicalPhase.FAILURE
        )

        raise RuntimeError(
            "Controlled cooperative release failed. "
            "Retreat was not started."
        )


    # ========================================================
    # RETREAT
    # ========================================================

    controller.set_phase(
        ClassicalPhase.RETREAT
    )


    print(
        "\n[RETREAT]"
    )


    (
        object_position_before_retreat,
        _,
        _,
        _,
    ) = object_state(
        model,
        data,
    )


    maximum_retreat_object_displacement = 0.0

    maximum_retreat_finger_normal = 0.0


    def observe_retreat(
        step_index,
    ):
        nonlocal maximum_retreat_object_displacement

        nonlocal maximum_retreat_finger_normal


        observe_global_state(
            step_index
        )


        (
            current_object_position,
            _rotation,
            _linear_velocity,
            _angular_velocity,
        ) = object_state(
            model,
            data,
        )


        (
            _finger_fz,
            _table_fz,
            finger_normal,
            _finger_contacts,
        ) = get_object_support_forces(
            model,
            data,
        )


        current_displacement = float(
            np.linalg.norm(
                current_object_position
                - object_position_before_retreat
            )
        )


        maximum_retreat_object_displacement = max(
            maximum_retreat_object_displacement,
            current_displacement,
        )


        maximum_retreat_finger_normal = max(
            maximum_retreat_finger_normal,
            finger_normal,
        )


    controller.execute_waypoint_trajectory(
        trajectories[
            "left_retreat_waypoints"
        ],
        trajectories[
            "right_retreat_waypoints"
        ],
        RETREAT_PHYSICS_STEPS_PER_WAYPOINT_SEGMENT,
        gripper_ctrl=OPEN_GRIPPER_CTRL,
        step_callback=observe_retreat,
        smooth_interpolation=True,
    )


    controller.hold_pose(
        trajectories[
            "left_retreat_q"
        ],
        trajectories[
            "right_retreat_q"
        ],
        POST_RETREAT_HOLD_STEPS,
        gripper_ctrl=OPEN_GRIPPER_CTRL,
        step_callback=observe_retreat,
    )


    mujoco.mj_forward(
        model,
        data,
    )


    (
        final_object_position,
        final_object_rotation,
        final_object_linear_velocity,
        final_object_angular_velocity,
    ) = object_state(
        model,
        data,
    )


    final_object_rpy = (
        rotation_matrix_to_rpy(
            final_object_rotation
        )
    )


    (
        final_finger_fz,
        final_table_fz,
        final_finger_normal,
        final_finger_contacts,
    ) = get_object_support_forces(
        model,
        data,
    )


    final_retreat_displacement = float(
        np.linalg.norm(
            final_object_position
            - object_position_before_retreat
        )
    )


    (
        final_goal_x_pass,
        final_goal_y_pass,
    ) = goal_containment(
        final_object_position
    )


    final_table_support_pass = (
        final_table_fz
        >= MIN_FINAL_TABLE_SUPPORT_N
    )


    final_finger_contact_pass = (
        final_finger_normal
        <= MAX_FINAL_FINGER_NORMAL_FORCE_N
    )


    final_linear_velocity_pass = (
        np.linalg.norm(
            final_object_linear_velocity
        )
        <= MAX_FINAL_LINEAR_VELOCITY_MPS
    )


    final_angular_velocity_pass = (
        np.linalg.norm(
            final_object_angular_velocity
        )
        <= MAX_FINAL_ANGULAR_VELOCITY_RADPS
    )


    final_roll_pitch_pass = all([
        abs(
            np.degrees(
                final_object_rpy[0]
            )
        )
        <= MAX_FINAL_ROLL_PITCH_DEG,

        abs(
            np.degrees(
                final_object_rpy[1]
            )
        )
        <= MAX_FINAL_ROLL_PITCH_DEG,
    ])


    final_retreat_displacement_pass = (
        final_retreat_displacement
        <= MAX_RETREAT_OBJECT_DISPLACEMENT_M
    )


    retreat_pass = all([
        final_goal_x_pass,
        final_goal_y_pass,
        final_table_support_pass,
        final_finger_contact_pass,
        final_linear_velocity_pass,
        final_angular_velocity_pass,
        final_roll_pitch_pass,
        final_retreat_displacement_pass,
    ])


    phase_results[
        "RETREAT"
    ] = retreat_pass


    print(
        "  final object position:",
        final_object_position,
    )

    print(
        "  final displacement [m]:",
        final_retreat_displacement,
    )

    print(
        "  maximum displacement [m]:",
        maximum_retreat_object_displacement,
    )

    print(
        "  maximum finger normal force [N]:",
        maximum_retreat_finger_normal,
    )

    print_phase_result(
        "RETREAT",
        retreat_pass,
    )


    # ========================================================
    # Final result
    # ========================================================

    complete_task_pass = all(
        phase_results.values()
    )


    if complete_task_pass:

        controller.set_phase(
            ClassicalPhase.SUCCESS
        )

    else:

        controller.set_phase(
            ClassicalPhase.FAILURE
        )


    baseline_wall_time = (
        time.perf_counter()
        - baseline_wall_start
    )


    simulated_time = float(
        data.time
    )


    final_goal_center_error_xy = float(
        np.linalg.norm(
            final_object_position[
                :2
            ]
            - np.array([
                GOAL_CENTER_X,
                GOAL_CENTER_Y,
            ])
        )
    )


    final_linear_speed = float(
        np.linalg.norm(
            final_object_linear_velocity
        )
    )


    final_angular_speed = float(
        np.linalg.norm(
            final_object_angular_velocity
        )
    )


    final_rpy_deg = np.degrees(
        final_object_rpy
    )


    # ========================================================
    # Compact benchmark report
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CLASSICAL BASELINE RESULT"
    )

    print(
        "=" * 70
    )


    for phase_name in [
        "APPROACH",
        "GRASP",
        "LIFT",
        "TRANSPORT",
        "PLACE",
        "RELEASE",
        "RETREAT",
    ]:

        print_phase_result(
            phase_name,
            phase_results[
                phase_name
            ],
        )


    print(
        "\nFinal state:"
    )

    print(
        "  object position [m]:",
        final_object_position,
    )

    print(
        "  object RPY [deg]:",
        final_rpy_deg,
    )

    print(
        "  goal-center XY error [m]:",
        final_goal_center_error_xy,
    )

    print(
        "  linear speed [m/s]:",
        final_linear_speed,
    )

    print(
        "  angular speed [rad/s]:",
        final_angular_speed,
    )


    print(
        "\nFinal support:"
    )

    print(
        "  table vertical support [N]:",
        final_table_fz,
    )

    print(
        "  finger vertical support [N]:",
        final_finger_fz,
    )

    print(
        "  finger normal-force sum [N]:",
        final_finger_normal,
    )

    print(
        "  finger contact count:",
        final_finger_contacts,
    )


    print(
        "\nRetreat:"
    )

    print(
        "  final object displacement [m]:",
        final_retreat_displacement,
    )

    print(
        "  maximum object displacement [m]:",
        maximum_retreat_object_displacement,
    )


    print(
        "\nTask-wide descriptive metrics:"
    )

    print(
        "  simulated time [s]:",
        simulated_time,
    )

    print(
        "  wall-clock execution time [s]:",
        baseline_wall_time,
    )

    print(
        "  max |roll/pitch| [deg]:",
        maximum_roll_pitch_deg,
    )

    print(
        "  max object linear speed [m/s]:",
        maximum_object_linear_speed,
    )

    print(
        "  max object angular speed [rad/s]:",
        maximum_object_angular_speed,
    )

    print(
        "  actuator-force-squared integral:",
        control_effort_integral,
    )

    print(
        "  joint-command variation integral:",
        command_variation_integral,
    )


    print(
        "\nFinal acceptance checks:"
    )

    print(
        "  inside goal X:",
        final_goal_x_pass,
    )

    print(
        "  inside goal Y:",
        final_goal_y_pass,
    )

    print(
        "  table support >=",
        MIN_FINAL_TABLE_SUPPORT_N,
        "N:",
        final_table_support_pass,
    )

    print(
        "  residual finger normal <=",
        MAX_FINAL_FINGER_NORMAL_FORCE_N,
        "N:",
        final_finger_contact_pass,
    )

    print(
        "  final linear speed <=",
        MAX_FINAL_LINEAR_VELOCITY_MPS,
        "m/s:",
        final_linear_velocity_pass,
    )

    print(
        "  final angular speed <=",
        MAX_FINAL_ANGULAR_VELOCITY_RADPS,
        "rad/s:",
        final_angular_velocity_pass,
    )

    print(
        "  final roll/pitch <=",
        MAX_FINAL_ROLL_PITCH_DEG,
        "deg:",
        final_roll_pitch_pass,
    )

    print(
        "  retreat displacement <=",
        MAX_RETREAT_OBJECT_DISPLACEMENT_M,
        "m:",
        final_retreat_displacement_pass,
    )


    print(
        "\n"
        + "=" * 70
    )


    if complete_task_pass:

        print(
            "CLASSICAL BASELINE: PASS"
        )

    else:

        print(
            "CLASSICAL BASELINE: FAIL"
        )


    print(
        "=" * 70
    )


    if not complete_task_pass:

        raise RuntimeError(
            "Final classical baseline acceptance "
            "criteria failed."
        )


    return {
        "success":
            complete_task_pass,

        "phase_results":
            phase_results,

        "simulated_time_s":
            simulated_time,

        "wall_time_s":
            baseline_wall_time,

        "final_object_position":
            final_object_position.copy(),

        "final_object_rpy_deg":
            final_rpy_deg.copy(),

        "goal_center_error_xy_m":
            final_goal_center_error_xy,

        "final_linear_speed_mps":
            final_linear_speed,

        "final_angular_speed_radps":
            final_angular_speed,

        "final_table_support_n":
            final_table_fz,

        "final_finger_normal_n":
            final_finger_normal,

        "final_finger_contacts":
            final_finger_contacts,

        "retreat_displacement_m":
            final_retreat_displacement,

        "max_retreat_displacement_m":
            maximum_retreat_object_displacement,

        "max_roll_pitch_deg":
            maximum_roll_pitch_deg,

        "max_object_linear_speed_mps":
            maximum_object_linear_speed,

        "max_object_angular_speed_radps":
            maximum_object_angular_speed,

        "control_effort_integral":
            control_effort_integral,

        "command_variation_integral":
            command_variation_integral,
    }


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_classical_baseline()
