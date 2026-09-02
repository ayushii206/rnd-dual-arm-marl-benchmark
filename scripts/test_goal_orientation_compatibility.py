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


from src.controllers.ik_controller import (
    solve_pose_ik,
    solve_position_ik,
)


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
# Geometry
# ============================================================

GRASP_Y = 0.145

GOAL_X = 0.45

NOMINAL_LIFT_Z = 0.58


LEFT_LIFT_POSITION = np.array([
    0.0,
    GRASP_Y,
    NOMINAL_LIFT_Z,
])


RIGHT_LIFT_POSITION = np.array([
    0.0,
    -GRASP_Y,
    NOMINAL_LIFT_Z,
])


# We already know X = 0.400 m is reachable.
# We use this valid configuration as the IK seed for the goal
# pose variants.
SEED_X = 0.40


LEFT_SEED_POSITION = np.array([
    SEED_X,
    GRASP_Y,
    NOMINAL_LIFT_Z,
])


RIGHT_SEED_POSITION = np.array([
    SEED_X,
    -GRASP_Y,
    NOMINAL_LIFT_Z,
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
# Acceptance tolerances
# ============================================================

POSITION_TOLERANCE_M = 1.0e-3

ORIENTATION_TOLERANCE_DEG = 1.0


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


def get_site_rotation(
    site_name,
):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id < 0:
        raise RuntimeError(
            f"Site not found: {site_name}"
        )

    return data.site_xmat[
        site_id
    ].reshape(
        3,
        3,
    ).copy()


def orientation_error_deg(
    target_rotation,
    achieved_rotation,
):
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

    angle_rad = np.arccos(
        cos_angle
    )

    return float(
        np.degrees(
            angle_rad
        )
    )

def relative_rotation_axis_angle(
    target_rotation,
    achieved_rotation,
):
    relative_rotation = (
        target_rotation.T
        @ achieved_rotation
    )

    cos_angle = np.clip(
        (
            np.trace(relative_rotation)
            - 1.0
        )
        / 2.0,
        -1.0,
        1.0,
    )

    angle = float(
        np.arccos(cos_angle)
    )

    if abs(angle) < 1.0e-8:
        axis = np.array([
            0.0,
            0.0,
            0.0,
        ])

        return (
            axis,
            0.0,
        )

    sin_angle = np.sin(angle)

    axis = np.array([
        relative_rotation[2, 1]
        - relative_rotation[1, 2],

        relative_rotation[0, 2]
        - relative_rotation[2, 0],

        relative_rotation[1, 0]
        - relative_rotation[0, 1],
    ])

    axis = (
        axis
        / (
            2.0
            * sin_angle
        )
    )

    axis_norm = np.linalg.norm(axis)

    if axis_norm > 1.0e-8:
        axis = (
            axis
            / axis_norm
        )

    return (
        axis,
        float(np.degrees(angle)),
    )

def rotation_axis_angle(
    rotation_matrix,
):
    cos_angle = np.clip(
        (
            np.trace(rotation_matrix)
            - 1.0
        )
        / 2.0,
        -1.0,
        1.0,
    )

    angle = float(
        np.arccos(cos_angle)
    )

    if abs(angle) < 1.0e-8:
        return (
            np.zeros(3),
            0.0,
        )

    sin_angle = np.sin(angle)

    axis = np.array([
        rotation_matrix[2, 1]
        - rotation_matrix[1, 2],

        rotation_matrix[0, 2]
        - rotation_matrix[2, 0],

        rotation_matrix[1, 0]
        - rotation_matrix[0, 1],
    ]) / (
        2.0
        * sin_angle
    )

    axis_norm = np.linalg.norm(
        axis
    )

    if axis_norm > 1.0e-8:
        axis = (
            axis
            / axis_norm
        )

    return (
        axis,
        float(
            np.degrees(angle)
        ),
    )
    
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
            lower
            <= joint_q
            <= upper
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


def minimum_joint_margin(
    rows,
):
    return min(
        row["margin"]
        for row in rows
    )


def all_joints_inside_limits(
    rows,
):
    return all(
        row["inside"]
        for row in rows
    )


def print_header(
    text,
):
    print(
        "\n"
        + "=" * 78
    )

    print(
        text
    )

    print(
        "=" * 78
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
# Get nominal EE orientations
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


# ============================================================
# First recover validated lifted configurations at X = 0
# ============================================================

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
# Recover validated X = 0.40 m configuration
#
# This becomes the initial guess for every goal variant.
# ============================================================

(
    left_seed_q,
    left_seed_achieved_position,
    left_seed_achieved_rotation,
) = solve_pose_ik(
    model=model,
    data=data,
    joint_names=left_joint_names,
    site_name=LEFT_EE_SITE,
    target_position=LEFT_SEED_POSITION,
    target_rotation=left_target_rotation,
    initial_q=left_lift_q,
)


(
    right_seed_q,
    right_seed_achieved_position,
    right_seed_achieved_rotation,
) = solve_pose_ik(
    model=model,
    data=data,
    joint_names=right_joint_names,
    site_name=RIGHT_EE_SITE,
    target_position=RIGHT_SEED_POSITION,
    target_rotation=right_target_rotation,
    initial_q=right_lift_q,
)


left_seed_error = float(
    np.linalg.norm(
        left_seed_achieved_position
        - LEFT_SEED_POSITION
    )
)

right_seed_error = float(
    np.linalg.norm(
        right_seed_achieved_position
        - RIGHT_SEED_POSITION
    )
)


print_header(
    "GOAL POSE VARIANT DIAGNOSTIC"
)


print(
    "\nActual goal X [m]:",
    GOAL_X,
)

print(
    "Grasp Y [m]:",
    GRASP_Y,
)

print(
    "Nominal lifted Z [m]:",
    NOMINAL_LIFT_Z,
)

print(
    "Validated seed X [m]:",
    SEED_X,
)


print_header(
    "VALIDATED X = 0.40 m SEED"
)


print(
    "Left seed position error [m]:",
    left_seed_error,
)

print(
    "Right seed position error [m]:",
    right_seed_error,
)


if (
    left_seed_error
    > POSITION_TOLERANCE_M
    or right_seed_error
    > POSITION_TOLERANCE_M
):
    raise RuntimeError(
        "The X = 0.40 m seed pose is not valid. "
        "Stop before testing goal variants."
    )


# ============================================================
# Variant definitions
# ============================================================

VARIANTS = [
    {
        "name": (
            "A - exact orientation, "
            "Z = 0.58 m"
        ),
        "mode": "pose",
        "z": 0.58,
    },
    {
        "name": (
            "B - exact orientation, "
            "Z = 0.56 m"
        ),
        "mode": "pose",
        "z": 0.56,
    },
    {
        "name": (
            "C - exact orientation, "
            "Z = 0.60 m"
        ),
        "mode": "pose",
        "z": 0.60,
    },
    {
        "name": (
            "D - position only, "
            "Z = 0.58 m"
        ),
        "mode": "position",
        "z": 0.58,
    },
]


# ============================================================
# Run variants
# ============================================================

results = []


for variant in VARIANTS:

    variant_name = (
        variant["name"]
    )

    mode = (
        variant["mode"]
    )

    target_z = float(
        variant["z"]
    )


    left_target_position = np.array([
        GOAL_X,
        GRASP_Y,
        target_z,
    ])


    right_target_position = np.array([
        GOAL_X,
        -GRASP_Y,
        target_z,
    ])


    print_header(
        variant_name
    )


    print(
        "\nLeft target position:",
        left_target_position,
    )

    print(
        "Right target position:",
        right_target_position,
    )


    # --------------------------------------------------------
    # Solve IK
    # --------------------------------------------------------

    if mode == "pose":

        (
            left_q,
            left_achieved_position,
            left_achieved_rotation,
        ) = solve_pose_ik(
            model=model,
            data=data,
            joint_names=left_joint_names,
            site_name=LEFT_EE_SITE,
            target_position=left_target_position,
            target_rotation=left_target_rotation,
            initial_q=left_seed_q,
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
            target_position=right_target_position,
            target_rotation=right_target_rotation,
            initial_q=right_seed_q,
        )


    elif mode == "position":

        (
            left_q,
            left_achieved_position,
            _,
        ) = solve_position_ik(
            model=model,
            data=data,
            joint_names=left_joint_names,
            site_name=LEFT_EE_SITE,
            target_position=left_target_position,
            initial_q=left_seed_q,
        )


        # Explicitly place solved joints into data so we can
        # measure the resulting EE orientation.
        set_arm_q(
            left_qpos_ids,
            left_q,
        )

        mujoco.mj_forward(
            model,
            data,
        )

        left_achieved_rotation = (
            get_site_rotation(
                LEFT_EE_SITE
            )
        )


        (
            right_q,
            right_achieved_position,
            _,
        ) = solve_position_ik(
            model=model,
            data=data,
            joint_names=right_joint_names,
            site_name=RIGHT_EE_SITE,
            target_position=right_target_position,
            initial_q=right_seed_q,
        )


        set_arm_q(
            right_qpos_ids,
            right_q,
        )

        mujoco.mj_forward(
            model,
            data,
        )

        right_achieved_rotation = (
            get_site_rotation(
                RIGHT_EE_SITE
            )
        )


    else:

        raise ValueError(
            f"Unknown mode: {mode}"
        )


    # --------------------------------------------------------
    # Position errors
    # --------------------------------------------------------

    left_position_residual = (
        left_achieved_position
        - left_target_position
    )

    right_position_residual = (
        right_achieved_position
        - right_target_position
    )


    left_position_error = float(
        np.linalg.norm(
            left_position_residual
        )
    )

    right_position_error = float(
        np.linalg.norm(
            right_position_residual
        )
    )


    # --------------------------------------------------------
    # Orientation deviations from nominal grasp orientation
    # --------------------------------------------------------

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
    
    left_world_rotation_delta = (
        left_achieved_rotation
        @ left_target_rotation.T
    )

    right_world_rotation_delta = (
        right_achieved_rotation
        @ right_target_rotation.T
    )
    
    world_rotation_difference = (
        left_world_rotation_delta.T
        @ right_world_rotation_delta
    )


    (
        world_rotation_difference_axis,
        world_rotation_difference_angle_deg,
    ) = rotation_axis_angle(
        world_rotation_difference
    )


    (
        left_world_rotation_axis,
        left_world_rotation_angle_deg,
    ) = rotation_axis_angle(
        left_world_rotation_delta
    )


    (
        right_world_rotation_axis,
        right_world_rotation_angle_deg,
    ) = rotation_axis_angle(
        right_world_rotation_delta
    )
    
    (
        left_relative_rotation_axis,
        left_relative_rotation_angle_deg,
    ) = relative_rotation_axis_angle(
        left_target_rotation,
        left_achieved_rotation,
    )


    (
        right_relative_rotation_axis,
        right_relative_rotation_angle_deg,
    ) = relative_rotation_axis_angle(
        right_target_rotation,
        right_achieved_rotation,
    )


    # --------------------------------------------------------
    # Joint limits
    # --------------------------------------------------------

    left_limit_report = (
        get_joint_limit_report(
            left_joint_names,
            left_q,
        )
    )

    right_limit_report = (
        get_joint_limit_report(
            right_joint_names,
            right_q,
        )
    )


    left_limits_pass = (
        all_joints_inside_limits(
            left_limit_report
        )
    )

    right_limits_pass = (
        all_joints_inside_limits(
            right_limit_report
        )
    )


    left_min_margin = (
        minimum_joint_margin(
            left_limit_report
        )
    )

    right_min_margin = (
        minimum_joint_margin(
            right_limit_report
        )
    )


    # --------------------------------------------------------
    # Acceptance
    # --------------------------------------------------------

    position_pass = (
        left_position_error
        <= POSITION_TOLERANCE_M
        and right_position_error
        <= POSITION_TOLERANCE_M
    )


    if mode == "pose":

        orientation_pass = (
            left_orientation_error
            <= ORIENTATION_TOLERANCE_DEG
            and right_orientation_error
            <= ORIENTATION_TOLERANCE_DEG
        )

        variant_pass = all([
            position_pass,
            orientation_pass,
            left_limits_pass,
            right_limits_pass,
        ])


    else:

        # Position-only mode deliberately does not enforce
        # orientation. We still print the orientation change
        # because a very large rotation may make the solution
        # unsuitable for maintaining the grasp.
        orientation_pass = None

        variant_pass = all([
            position_pass,
            left_limits_pass,
            right_limits_pass,
        ])


    # --------------------------------------------------------
    # Print detailed result
    # --------------------------------------------------------

    print(
        "\nLEFT"
    )

    print(
        "Achieved position:",
        left_achieved_position,
    )

    print(
        "Residual [m]:",
        left_position_residual,
    )

    print(
        "Position error norm [m]:",
        left_position_error,
    )

    print(
        "Orientation deviation from nominal [deg]:",
        left_orientation_error,
    )
    
    print(
        "Relative rotation axis "
        "[nominal EE frame]:",
        left_relative_rotation_axis,
    )
    
    print(
        "Rotation axis [world frame]:",
        left_world_rotation_axis,
    )

    print(
        "Rotation angle [world frame, deg]:",
        left_world_rotation_angle_deg,
    )

    print(
        "Minimum joint-limit margin [rad]:",
        left_min_margin,
    )

    print(
        "All joints inside limits:",
        left_limits_pass,
    )

    print(
        "q:",
        left_q,
    )


    print(
        "\nRIGHT"
    )

    print(
        "Achieved position:",
        right_achieved_position,
    )

    print(
        "Residual [m]:",
        right_position_residual,
    )

    print(
        "Position error norm [m]:",
        right_position_error,
    )

    print(
        "Orientation deviation from nominal [deg]:",
        right_orientation_error,
    )
    
    print(
        "Relative rotation axis "
        "[nominal EE frame]:",
        right_relative_rotation_axis,
    )
    
    print(
        "Rotation axis [world frame]:",
        right_world_rotation_axis,
    )

    print(
        "Rotation angle [world frame, deg]:",
        right_world_rotation_angle_deg,
    )

    print(
        "Minimum joint-limit margin [rad]:",
        right_min_margin,
    )

    print(
        "All joints inside limits:",
        right_limits_pass,
    )

    print(
        "q:",
        right_q,
    )


    print(
        "\nPosition criterion passed:",
        position_pass,
    )


    if mode == "pose":

        print(
            "Orientation criterion passed:",
            orientation_pass,
        )

    else:

        print(
            "Orientation criterion:",
            "NOT ENFORCED",
        )


    print(
        "Variant result:",
        variant_pass,
    )


    results.append({
        "name": variant_name,
        "mode": mode,
        "z": target_z,
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
        "left_min_margin": (
            left_min_margin
        ),
        "right_min_margin": (
            right_min_margin
        ),
        "pass": variant_pass,
    })
    
    print(
        "\nLeft-vs-right world rotation difference [deg]:",
        world_rotation_difference_angle_deg,
    )


# ============================================================
# Summary table
# ============================================================

print_header(
    "GOAL POSE VARIANT SUMMARY"
)


print(
    "Variant | Mode | Z | "
    "L pos err mm | R pos err mm | "
    "L ori deg | R ori deg | PASS"
)

print(
    "-" * 105
)


for result in results:

    print(
        f"{result['name']:<36} | "
        f"{result['mode']:<8} | "
        f"{result['z']:.3f} | "
        f"{result['left_position_error'] * 1000:12.4f} | "
        f"{result['right_position_error'] * 1000:12.4f} | "
        f"{result['left_orientation_error']:9.4f} | "
        f"{result['right_orientation_error']:9.4f} | "
        f"{result['pass']}"
    )


print_header(
    "INTERPRETATION FLAGS"
)


exact_nominal_pass = (
    results[0]["pass"]
)

lower_z_pass = (
    results[1]["pass"]
)

higher_z_pass = (
    results[2]["pass"]
)

position_only_pass = (
    results[3]["pass"]
)


print(
    "Exact orientation at Z=0.58 passed:",
    exact_nominal_pass,
)

print(
    "Exact orientation at Z=0.56 passed:",
    lower_z_pass,
)

print(
    "Exact orientation at Z=0.60 passed:",
    higher_z_pass,
)

print(
    "Position-only IK at Z=0.58 passed:",
    position_only_pass,
)


print(
    "\n"
    + "=" * 78
)


if position_only_pass and not exact_nominal_pass:

    print(
        "DIAGNOSTIC RESULT:"
    )

    print(
        "X = 0.45 m is positionally reachable, "
        "but the exact nominal EE orientation "
        "appears to restrict the workspace."
    )


elif (
    lower_z_pass
    or higher_z_pass
):

    print(
        "DIAGNOSTIC RESULT:"
    )

    print(
        "Changing transport height improves "
        "goal-pose reachability."
    )


elif exact_nominal_pass:

    print(
        "DIAGNOSTIC RESULT:"
    )

    print(
        "The nominal goal pose is reachable "
        "in this diagnostic."
    )


else:

    print(
        "DIAGNOSTIC RESULT:"
    )

    print(
        "The tested goal pose variants did not "
        "recover X = 0.45 m within the current "
        "acceptance criteria."
    )


print(
    "=" * 78
)
