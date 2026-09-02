from pathlib import Path
import sys

import mujoco
import numpy as np


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.controllers.ik_controller import solve_pose_ik


# ============================================================
# Model
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "models"
    / "dual_panda"
    / "transport_scene.xml"
)


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    model
)


# ============================================================
# Robot names
# ============================================================

left_joint_names = [
    f"left_joint{i}"
    for i in range(1, 8)
]

right_joint_names = [
    f"right_joint{i}"
    for i in range(1, 8)
]


LEFT_EE_SITE = "left_ee_site"
RIGHT_EE_SITE = "right_ee_site"


# ============================================================
# Validated geometry
# ============================================================

GRASP_Y = 0.145

LIFTED_Z = 0.58

GOAL_X = 0.45


LEFT_GOAL_POSITION = np.array([
    GOAL_X,
    GRASP_Y,
    LIFTED_Z,
])


RIGHT_GOAL_POSITION = np.array([
    GOAL_X,
    -GRASP_Y,
    LIFTED_Z,
])


# ============================================================
# Nominal arm configuration
# ============================================================

INITIAL_Q = np.array([
    0.0,
    -0.4,
    0.0,
    -1.8,
    0.0,
    1.4,
    0.8,
])


# ============================================================
# Helpers
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

        if joint_id < 0:
            raise RuntimeError(
                f"Joint not found: {joint_name}"
            )

        qpos_ids.append(
            model.jnt_qposadr[
                joint_id
            ]
        )

    return np.array(
        qpos_ids,
        dtype=int,
    )


def set_arm_q(
    qpos_ids,
    q,
):
    data.qpos[
        qpos_ids
    ] = q


def get_site_position(
    site_name,
):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    return data.site_xpos[
        site_id
    ].copy()


def get_site_rotation(
    site_name,
):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    return data.site_xmat[
        site_id
    ].reshape(
        3,
        3,
    ).copy()


def get_joint_limit_report(
    joint_names,
    q,
):
    rows = []

    for joint_name, joint_q in zip(
        joint_names,
        q,
    ):

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        lower = float(
            model.jnt_range[
                joint_id,
                0,
            ]
        )

        upper = float(
            model.jnt_range[
                joint_id,
                1,
            ]
        )

        lower_margin = (
            float(joint_q)
            - lower
        )

        upper_margin = (
            upper
            - float(joint_q)
        )

        nearest_margin = min(
            lower_margin,
            upper_margin,
        )

        inside_limits = (
            lower <= joint_q <= upper
        )

        rows.append({
            "joint": joint_name,
            "q": float(joint_q),
            "lower": lower,
            "upper": upper,
            "margin": nearest_margin,
            "inside": inside_limits,
        })

    return rows


def print_joint_limit_report(
    title,
    rows,
):
    print(
        "\n"
        + "=" * 85
    )

    print(
        title
    )

    print(
        "=" * 85
    )

    print(
        "joint                q         lower       upper       nearest_margin     inside"
    )

    print(
        "-" * 85
    )

    for row in rows:

        print(
            f"{row['joint']:<18}"
            f"{row['q']:+10.6f}  "
            f"{row['lower']:+10.6f}  "
            f"{row['upper']:+10.6f}  "
            f"{row['margin']:+14.6f}     "
            f"{row['inside']}"
        )


# ============================================================
# Joint qpos addresses
# ============================================================

left_qpos_ids = get_qpos_ids(
    left_joint_names
)

right_qpos_ids = get_qpos_ids(
    right_joint_names
)


# ============================================================
# Capture the validated nominal wrist orientations
# ============================================================

mujoco.mj_resetData(
    model,
    data,
)


set_arm_q(
    left_qpos_ids,
    INITIAL_Q,
)

set_arm_q(
    right_qpos_ids,
    INITIAL_Q,
)


mujoco.mj_forward(
    model,
    data,
)


left_target_rotation = (
    get_site_rotation(
        LEFT_EE_SITE
    )
)

right_target_rotation = (
    get_site_rotation(
        RIGHT_EE_SITE
    )
)


print(
    "=" * 70
)

print(
    "GOAL-SIDE DUAL-ARM REACHABILITY TEST"
)

print(
    "=" * 70
)

print(
    "\nGoal object X:",
    GOAL_X,
)

print(
    "Lifted object Z:",
    LIFTED_Z,
)

print(
    "Left goal EE target:",
    LEFT_GOAL_POSITION,
)

print(
    "Right goal EE target:",
    RIGHT_GOAL_POSITION,
)


# ============================================================
# First solve the already validated lifted pose
# ============================================================

LEFT_LIFT_POSITION = np.array([
    0.0,
    GRASP_Y,
    LIFTED_Z,
])


RIGHT_LIFT_POSITION = np.array([
    0.0,
    -GRASP_Y,
    LIFTED_Z,
])


left_lift_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=left_joint_names,
    site_name=LEFT_EE_SITE,
    target_position=LEFT_LIFT_POSITION,
    target_rotation=left_target_rotation,
    initial_q=INITIAL_Q,
)


right_lift_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=right_joint_names,
    site_name=RIGHT_EE_SITE,
    target_position=RIGHT_LIFT_POSITION,
    target_rotation=right_target_rotation,
    initial_q=INITIAL_Q,
)


# ============================================================
# Sequential reachability waypoints
#
# We use continuation rather than asking IK to jump directly
# from X = 0.00 m to X = 0.45 m.
# ============================================================

WAYPOINT_X_VALUES = [
    0.20,
    0.30,
    0.40,
    0.402,
    0.404,
    0.406,
    0.408,
    0.410,
    0.420,
    0.430,
    0.440,
    0.450,
]


left_previous_q = (
    left_lift_q.copy()
)

right_previous_q = (
    right_lift_q.copy()
)


left_goal_q = None
right_goal_q = None


all_waypoints_pass = True

furthest_reachable_x = 0.0

for waypoint_x in WAYPOINT_X_VALUES:

    left_position = np.array([
        waypoint_x,
        GRASP_Y,
        LIFTED_Z,
    ])

    right_position = np.array([
        waypoint_x,
        -GRASP_Y,
        LIFTED_Z,
    ])


    (
        left_q,
        left_achieved_position,
        left_achieved_rotation,
    ) = solve_pose_ik(
        model=model,
        data=data,
        joint_names=left_joint_names,
        site_name=LEFT_EE_SITE,
        target_position=left_position,
        target_rotation=left_target_rotation,
        initial_q=left_previous_q,
    )


    (
        right_q,
        right_achieved_position,
        right_achieved_rotation,
    ) = solve_pose_ik(
        model=model,
        data=data,
        joint_names=right_joint_names,
        site_name=RIGHT_EE_SITE,
        target_position=right_position,
        target_rotation=right_target_rotation,
        initial_q=right_previous_q,
    )


    left_residual = (
        left_achieved_position
        - left_position
    )

    right_residual = (
        right_achieved_position
        - right_position
    )
    
    left_rotation_difference = (
        left_target_rotation.T
        @ left_achieved_rotation
    )

    right_rotation_difference = (
        right_target_rotation.T
        @ right_achieved_rotation
    )


    left_orientation_cos_angle = np.clip(
        (
            np.trace(
                left_rotation_difference
            )
            - 1.0
        )
        / 2.0,
        -1.0,
        1.0,
    )

    right_orientation_cos_angle = np.clip(
        (
            np.trace(
                right_rotation_difference
            )
            - 1.0
        )
        / 2.0,
        -1.0,
        1.0,
    )


    left_orientation_error_rad = float(
        np.arccos(
            left_orientation_cos_angle
        )
    )

    right_orientation_error_rad = float(
        np.arccos(
            right_orientation_cos_angle
        )
    )

    left_error_norm = float(
        np.linalg.norm(
            left_residual
        )
    )

    right_error_norm = float(
        np.linalg.norm(
            right_residual
        )
    )


    POSITION_TOLERANCE = 1.0e-3
    ORIENTATION_TOLERANCE_DEG = 1.0


    left_waypoint_pass = (
        left_error_norm
        <= POSITION_TOLERANCE
        and np.degrees(
            left_orientation_error_rad
        )
        <= ORIENTATION_TOLERANCE_DEG
    )


    right_waypoint_pass = (
        right_error_norm
        <= POSITION_TOLERANCE
        and np.degrees(
            right_orientation_error_rad
        )
        <= ORIENTATION_TOLERANCE_DEG
    )


    waypoint_pass = (
        left_waypoint_pass
        and right_waypoint_pass
    )
    
    if waypoint_pass:

        furthest_reachable_x = (
            waypoint_x
        )

    if not waypoint_pass:
        all_waypoints_pass = False


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"WAYPOINT X = {waypoint_x:.3f} m"
    )

    print(
        "=" * 70
    )


    print(
        "\nLEFT"
    )

    print(
        "Target position:",
        left_position,
    )

    print(
        "Achieved position:",
        left_achieved_position,
    )

    print(
        "Residual [m]:",
        left_residual,
    )

    print(
        "Position error norm [m]:",
        left_error_norm,
    )
    
    print(
        "Orientation error [deg]:",
        np.degrees(
            left_orientation_error_rad
        ),
    )

    print(
        "q:",
        left_q,
    )


    print(
        "\nRIGHT"
    )

    print(
        "Target position:",
        right_position,
    )

    print(
        "Achieved position:",
        right_achieved_position,
    )

    print(
        "Residual [m]:",
        right_residual,
    )

    print(
        "Position error norm [m]:",
        right_error_norm,
    )
    
    print(
        "Orientation error [deg]:",
        np.degrees(
            right_orientation_error_rad
        ),
    )

    print(
        "q:",
        right_q,
    )


    print(
        "\nWaypoint pass:",
        waypoint_pass,
    )


    if waypoint_pass:

        left_previous_q = (
            left_q.copy()
        )

        right_previous_q = (
            right_q.copy()
        )


    if np.isclose(
        waypoint_x,
        GOAL_X,
    ):

        left_goal_q = (
            left_q.copy()
        )

        right_goal_q = (
            right_q.copy()
        )


# ============================================================
# Goal joint-limit analysis
# ============================================================

if (
    left_goal_q is None
    or right_goal_q is None
):
    raise RuntimeError(
        "Goal configuration was not generated."
    )


left_limit_report = (
    get_joint_limit_report(
        left_joint_names,
        left_goal_q,
    )
)

right_limit_report = (
    get_joint_limit_report(
        right_joint_names,
        right_goal_q,
    )
)


print_joint_limit_report(
    "LEFT ARM JOINT LIMIT REPORT AT GOAL",
    left_limit_report,
)

print_joint_limit_report(
    "RIGHT ARM JOINT LIMIT REPORT AT GOAL",
    right_limit_report,
)


left_limits_pass = all(
    row["inside"]
    for row in left_limit_report
)

right_limits_pass = all(
    row["inside"]
    for row in right_limit_report
)


left_minimum_margin = min(
    row["margin"]
    for row in left_limit_report
)

right_minimum_margin = min(
    row["margin"]
    for row in right_limit_report
)


# ============================================================
# Symmetry diagnostic
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FINAL GOAL CONFIGURATION"
)

print(
    "=" * 70
)


print(
    "\nLeft goal q:",
    left_goal_q,
)

print(
    "Right goal q:",
    right_goal_q,
)


print(
    "\nLeft minimum joint-limit margin [rad]:",
    left_minimum_margin,
)

print(
    "Right minimum joint-limit margin [rad]:",
    right_minimum_margin,
)


print(
    "\nAll waypoint IK solutions passed:",
    all_waypoints_pass,
)

print(
    "Left arm inside all joint limits:",
    left_limits_pass,
)

print(
    "Right arm inside all joint limits:",
    right_limits_pass,
)


goal_reachability_pass = all([
    all_waypoints_pass,
    left_limits_pass,
    right_limits_pass,
])


print(
    "\n"
    + "=" * 70
)

print(
    "\nFurthest validated lifted X position [m]:",
    furthest_reachable_x,
)

print(
    "GOAL REACHABILITY RESULT"
)

print(
    "=" * 70
)


if goal_reachability_pass:

    print(
        "RESULT: PASS"
    )

    print(
        "Both arms can kinematically reach "
        "the lifted goal-side grasp pose."
    )

else:

    print(
        "RESULT: FAIL"
    )

    print(
        "Goal-side grasp pose is not yet "
        "kinematically validated."
    )


print(
    "=" * 70
)
