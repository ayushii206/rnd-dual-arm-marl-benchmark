from pathlib import Path
import sys

import mujoco
import numpy as np


# ============================================================
# Project import setup
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

LEFT_JOINT_NAMES = [
    f"left_joint{i}"
    for i in range(1, 8)
]

RIGHT_JOINT_NAMES = [
    f"right_joint{i}"
    for i in range(1, 8)
]


LEFT_EE_SITE = "left_ee_site"
RIGHT_EE_SITE = "right_ee_site"


# ============================================================
# Task geometry
# ============================================================

GRASP_Y = 0.145

GOAL_X = 0.45

LIFTED_Z = 0.58


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


SEED_X = 0.40


LEFT_SEED_POSITION = np.array([
    SEED_X,
    GRASP_Y,
    LIFTED_Z,
])


RIGHT_SEED_POSITION = np.array([
    SEED_X,
    -GRASP_Y,
    LIFTED_Z,
])


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
# Nominal configuration
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
# Acceptance criteria
# ============================================================

POSITION_TOLERANCE_M = 1.0e-3

ORIENTATION_TOLERANCE_DEG = 1.0


# ============================================================
# Common orientation scan
#
# Both arms receive EXACTLY THE SAME rotation in world space.
#
# Positive and negative angles are included because we do not
# want to assume the useful direction from sign conventions.
# ============================================================

COMMON_Y_ROTATION_ANGLES_DEG = [
    -8.0,
    -8.1,
    -8.2,
    -8.3,
    -8.4,
    -8.5,
    -8.6,
    -8.7,
    -8.8,
    -8.9,
    -9.0,
]


# ============================================================
# Helpers
# ============================================================

def get_qpos_ids(
    joint_names,
):
    ids = []

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

        ids.append(
            model.jnt_qposadr[
                joint_id
            ]
        )

    return np.array(
        ids,
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


def rotation_y_world(
    angle_deg,
):
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
    difference = (
        target_rotation.T
        @ achieved_rotation
    )

    cos_angle = np.clip(
        (
            np.trace(
                difference
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


def get_joint_limit_margin(
    joint_names,
    q,
):
    margins = []

    all_inside = True

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

        joint_q = float(
            joint_q
        )

        inside = (
            lower
            <= joint_q
            <= upper
        )

        all_inside = (
            all_inside
            and inside
        )

        margins.append(
            min(
                joint_q - lower,
                upper - joint_q,
            )
        )

    return (
        all_inside,
        min(margins),
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
# Joint indices
# ============================================================

left_qpos_ids = get_qpos_ids(
    LEFT_JOINT_NAMES
)

right_qpos_ids = get_qpos_ids(
    RIGHT_JOINT_NAMES
)


# ============================================================
# Nominal orientations
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


left_nominal_rotation = (
    get_site_rotation(
        LEFT_EE_SITE
    )
)

right_nominal_rotation = (
    get_site_rotation(
        RIGHT_EE_SITE
    )
)


# ============================================================
# Recover validated lifted configuration
# ============================================================

(
    left_lift_q,
    _,
    _,
) = solve_pose_ik(
    model=model,
    data=data,
    joint_names=LEFT_JOINT_NAMES,
    site_name=LEFT_EE_SITE,
    target_position=LEFT_LIFT_POSITION,
    target_rotation=left_nominal_rotation,
    initial_q=INITIAL_Q,
)


(
    right_lift_q,
    _,
    _,
) = solve_pose_ik(
    model=model,
    data=data,
    joint_names=RIGHT_JOINT_NAMES,
    site_name=RIGHT_EE_SITE,
    target_position=RIGHT_LIFT_POSITION,
    target_rotation=right_nominal_rotation,
    initial_q=INITIAL_Q,
)


# ============================================================
# Recover validated X = 0.40 configuration
# ============================================================

(
    left_seed_q,
    left_seed_position,
    left_seed_rotation,
) = solve_pose_ik(
    model=model,
    data=data,
    joint_names=LEFT_JOINT_NAMES,
    site_name=LEFT_EE_SITE,
    target_position=LEFT_SEED_POSITION,
    target_rotation=left_nominal_rotation,
    initial_q=left_lift_q,
)


(
    right_seed_q,
    right_seed_position,
    right_seed_rotation,
) = solve_pose_ik(
    model=model,
    data=data,
    joint_names=RIGHT_JOINT_NAMES,
    site_name=RIGHT_EE_SITE,
    target_position=RIGHT_SEED_POSITION,
    target_rotation=right_nominal_rotation,
    initial_q=right_lift_q,
)


left_seed_error = np.linalg.norm(
    left_seed_position
    - LEFT_SEED_POSITION
)

right_seed_error = np.linalg.norm(
    right_seed_position
    - RIGHT_SEED_POSITION
)


print_header(
    "COMMON GOAL ORIENTATION DIAGNOSTIC"
)


print(
    "Goal X [m]:",
    GOAL_X,
)

print(
    "Goal Z [m]:",
    LIFTED_Z,
)

print(
    "Left seed error [mm]:",
    left_seed_error * 1000.0,
)

print(
    "Right seed error [mm]:",
    right_seed_error * 1000.0,
)


if (
    left_seed_error
    > POSITION_TOLERANCE_M
    or right_seed_error
    > POSITION_TOLERANCE_M
):
    raise RuntimeError(
        "Validated X=0.40 seed could not be recovered."
    )


# ============================================================
# Scan common world-Y rotations
# ============================================================

results = []


for angle_deg in COMMON_Y_ROTATION_ANGLES_DEG:

    common_rotation = rotation_y_world(
        angle_deg
    )


    # IMPORTANT:
    #
    # Premultiplication means the SAME WORLD-FRAME rotation
    # is applied to both nominal end-effector orientations.
    #
    # This preserves the relative orientation between the
    # two grippers.
    left_target_rotation = (
        common_rotation
        @ left_nominal_rotation
    )

    right_target_rotation = (
        common_rotation
        @ right_nominal_rotation
    )


    (
        left_q,
        left_achieved_position,
        left_achieved_rotation,
    ) = solve_pose_ik(
        model=model,
        data=data,
        joint_names=LEFT_JOINT_NAMES,
        site_name=LEFT_EE_SITE,
        target_position=LEFT_GOAL_POSITION,
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
        joint_names=RIGHT_JOINT_NAMES,
        site_name=RIGHT_EE_SITE,
        target_position=RIGHT_GOAL_POSITION,
        target_rotation=right_target_rotation,
        initial_q=right_seed_q,
    )


    left_position_error = float(
        np.linalg.norm(
            left_achieved_position
            - LEFT_GOAL_POSITION
        )
    )

    right_position_error = float(
        np.linalg.norm(
            right_achieved_position
            - RIGHT_GOAL_POSITION
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


    (
        left_limits_inside,
        left_min_margin,
    ) = get_joint_limit_margin(
        LEFT_JOINT_NAMES,
        left_q,
    )


    (
        right_limits_inside,
        right_min_margin,
    ) = get_joint_limit_margin(
        RIGHT_JOINT_NAMES,
        right_q,
    )


    position_pass = (
        left_position_error
        <= POSITION_TOLERANCE_M
        and right_position_error
        <= POSITION_TOLERANCE_M
    )


    orientation_pass = (
        left_orientation_error
        <= ORIENTATION_TOLERANCE_DEG
        and right_orientation_error
        <= ORIENTATION_TOLERANCE_DEG
    )


    pass_result = all([
        position_pass,
        orientation_pass,
        left_limits_inside,
        right_limits_inside,
    ])


    results.append({
        "angle": angle_deg,
        "left_pos_mm": (
            left_position_error
            * 1000.0
        ),
        "right_pos_mm": (
            right_position_error
            * 1000.0
        ),
        "left_ori_deg": (
            left_orientation_error
        ),
        "right_ori_deg": (
            right_orientation_error
        ),
        "left_margin": (
            left_min_margin
        ),
        "right_margin": (
            right_min_margin
        ),
        "pass": pass_result,
        "left_q": left_q,
        "right_q": right_q,
    })


# ============================================================
# Print summary
# ============================================================

print_header(
    "COMMON WORLD-Y ROTATION SUMMARY"
)


print(
    "Angle deg | "
    "L pos mm | "
    "R pos mm | "
    "L ori deg | "
    "R ori deg | "
    "L margin | "
    "R margin | "
    "PASS"
)


print(
    "-" * 104
)


for result in results:

    print(
        f"{result['angle']:9.2f} | "
        f"{result['left_pos_mm']:8.3f} | "
        f"{result['right_pos_mm']:8.3f} | "
        f"{result['left_ori_deg']:9.4f} | "
        f"{result['right_ori_deg']:9.4f} | "
        f"{result['left_margin']:8.4f} | "
        f"{result['right_margin']:8.4f} | "
        f"{result['pass']}"
    )


# ============================================================
# Find smallest successful common rotation
# ============================================================

successful_results = [
    result
    for result in results
    if result["pass"]
]


print_header(
    "RESULT"
)


if successful_results:

    best_result = min(
        successful_results,
        key=lambda result: abs(
            result["angle"]
        ),
    )


    print(
        "PASS"
    )

    print(
        "Smallest successful common "
        "world-Y rotation [deg]:",
        best_result["angle"],
    )

    print(
        "Left position error [mm]:",
        best_result["left_pos_mm"],
    )

    print(
        "Right position error [mm]:",
        best_result["right_pos_mm"],
    )

    print(
        "Left orientation residual [deg]:",
        best_result["left_ori_deg"],
    )

    print(
        "Right orientation residual [deg]:",
        best_result["right_ori_deg"],
    )

    print(
        "Left minimum joint margin [rad]:",
        best_result["left_margin"],
    )

    print(
        "Right minimum joint margin [rad]:",
        best_result["right_margin"],
    )

    print(
        "\nLeft goal q:"
    )

    print(
        best_result["left_q"]
    )

    print(
        "\nRight goal q:"
    )

    print(
        best_result["right_q"]
    )


else:

    print(
        "FAIL"
    )

    print(
        "No tested common world-Y rotation "
        "made the X=0.45 m goal pose reachable "
        "within the current tolerances."
    )
