import numpy as np
import mujoco

from src.controllers.ik_controller import solve_pose_ik


# ============================================================
# Model / task configuration
# ============================================================

MODEL_PATH = (
    "assets/models/dual_panda/transport_scene.xml"
)

GOAL_X = 0.45
GOAL_REFERENCE_Z = 0.58

GRASP_Y = 0.145

POSITION_TOLERANCE_M = 1.0e-3
ORIENTATION_TOLERANCE_DEG = 1.0


# ============================================================
# Object geometry
# ============================================================

# Main transport box half-dimensions.
OBJECT_HALF_X = 0.20
OBJECT_HALF_Y = 0.12
OBJECT_HALF_Z = 0.06

TABLE_TOP_Z = 0.50


# From the validated stable lift:
#
# EE midpoint/reference Z = 0.580000 m
# settled object center Z ≈ 0.573985 m
#
# Therefore the object center sits approximately 6.015 mm
# below the bilateral EE midpoint.
OBJECT_CENTER_OFFSET_FROM_REFERENCE = np.array([
    0.0,
    0.0,
    -0.006015,
])


# We do not want a solution that merely touches the table.
# This is used only to rank candidates geometrically.
CLEARANCE_SAFETY_MARGIN_M = 0.002


# ============================================================
# Search ranges
# ============================================================

# Coarse search first.
#
# Pitch contains the previously successful -9/-10 degree area.
# Roll/yaw allow the rigid object orientation to change in 3D.

ROLL_VALUES_DEG = np.arange(
    -8.0,
    8.01,
    2.0,
)

PITCH_VALUES_DEG = np.arange(
    -12.0,
    0.01,
    1.0,
)

YAW_VALUES_DEG = np.arange(
    -8.0,
    8.01,
    2.0,
)


# Number of successful candidates to display in detail.
TOP_CANDIDATE_COUNT = 30


# ============================================================
# Panda configuration
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


LEFT_JOINT_NAMES = [
    f"left_joint{i}"
    for i in range(1, 8)
]

RIGHT_JOINT_NAMES = [
    f"right_joint{i}"
    for i in range(1, 8)
]


LEFT_LIFT_POSITION = np.array([
    0.0,
    GRASP_Y,
    0.58,
])

RIGHT_LIFT_POSITION = np.array([
    0.0,
    -GRASP_Y,
    0.58,
])


# We use the already-known reachable X=0.40 configuration as
# the seed for the goal-pose orientation search.
#
# This avoids asking the IK solver to jump directly from the
# original lift pose to X=0.45 for every candidate.

SEED_X = 0.40

LEFT_SEED_POSITION = np.array([
    SEED_X,
    GRASP_Y,
    GOAL_REFERENCE_Z,
])

RIGHT_SEED_POSITION = np.array([
    SEED_X,
    -GRASP_Y,
    GOAL_REFERENCE_Z,
])


LEFT_GOAL_POSITION = np.array([
    GOAL_X,
    GRASP_Y,
    GOAL_REFERENCE_Z,
])

RIGHT_GOAL_POSITION = np.array([
    GOAL_X,
    -GRASP_Y,
    GOAL_REFERENCE_Z,
])


# ============================================================
# Helpers
# ============================================================

def get_qpos_ids(
    model,
    joint_names,
):
    qpos_ids = []

    for joint_name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_ids.append(
            model.jnt_qposadr[
                joint_id
            ]
        )

    return qpos_ids


def set_arm_q(
    data,
    qpos_ids,
    q,
):
    data.qpos[
        qpos_ids
    ] = q


def site_rotation(
    model,
    data,
    site_name,
):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    return (
        data.site_xmat[
            site_id
        ]
        .reshape(
            3,
            3,
        )
        .copy()
    )


def orientation_error_deg(
    target_rotation,
    achieved_rotation,
):
    rotation_error = (
        target_rotation.T
        @ achieved_rotation
    )

    cos_angle = (
        np.trace(rotation_error) - 1.0
    ) / 2.0

    cos_angle = np.clip(
        cos_angle,
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


def rotation_x_world(
    angle_deg,
):
    angle_rad = np.radians(
        angle_deg
    )

    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def rotation_y_world(
    angle_deg,
):
    angle_rad = np.radians(
        angle_deg
    )

    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rotation_z_world(
    angle_deg,
):
    angle_rad = np.radians(
        angle_deg
    )

    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def common_world_rotation(
    roll_deg,
    pitch_deg,
    yaw_deg,
):
    """
    World-frame roll/pitch/yaw composition.

    The rotation is applied to the nominal gripper orientations as:

        R_target = R_common @ R_nominal

    Composition:
        R_common = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """

    return (
        rotation_z_world(yaw_deg)
        @ rotation_y_world(pitch_deg)
        @ rotation_x_world(roll_deg)
    )


def object_vertical_half_extent(
    object_rotation,
):
    """
    Vertical half-extent of an oriented box.

    For an oriented box with half-sizes hx, hy, hz:

        world Z extent
        =
        |R[2,0]| hx
        + |R[2,1]| hy
        + |R[2,2]| hz
    """

    return float(
        abs(object_rotation[2, 0])
        * OBJECT_HALF_X
        +
        abs(object_rotation[2, 1])
        * OBJECT_HALF_Y
        +
        abs(object_rotation[2, 2])
        * OBJECT_HALF_Z
    )


def geometric_clearance_metrics(
    common_rotation,
):
    """
    Estimate object/table clearance assuming the object follows
    the common rigid-body rotation.

    The object's center is slightly below the bilateral EE
    reference point in the validated grasp.
    """

    rotated_center_offset = (
        common_rotation
        @ OBJECT_CENTER_OFFSET_FROM_REFERENCE
    )

    center_offset_world_z = float(
        rotated_center_offset[2]
    )

    half_extent_z = (
        object_vertical_half_extent(
            common_rotation
        )
    )

    object_center_z_at_test_height = (
        GOAL_REFERENCE_Z
        + center_offset_world_z
    )

    object_bottom_z_at_test_height = (
        object_center_z_at_test_height
        - half_extent_z
    )

    clearance_at_test_height = (
        object_bottom_z_at_test_height
        - TABLE_TOP_Z
    )

    required_reference_z_for_touch = (
        TABLE_TOP_Z
        + half_extent_z
        - center_offset_world_z
    )

    required_reference_z_with_margin = (
        required_reference_z_for_touch
        + CLEARANCE_SAFETY_MARGIN_M
    )

    return {
        "half_extent_z": (
            half_extent_z
        ),
        "center_offset_world_z": (
            center_offset_world_z
        ),
        "bottom_z_at_test_height": (
            object_bottom_z_at_test_height
        ),
        "clearance_at_test_height": (
            clearance_at_test_height
        ),
        "required_reference_z_for_touch": (
            required_reference_z_for_touch
        ),
        "required_reference_z_with_margin": (
            required_reference_z_with_margin
        ),
    }


# ============================================================
# Load model
# ============================================================

model = mujoco.MjModel.from_xml_path(
    MODEL_PATH
)

data = mujoco.MjData(
    model
)


left_qpos_ids = get_qpos_ids(
    model,
    LEFT_JOINT_NAMES,
)

right_qpos_ids = get_qpos_ids(
    model,
    RIGHT_JOINT_NAMES,
)


# ============================================================
# Capture nominal EE orientations
# ============================================================

set_arm_q(
    data,
    left_qpos_ids,
    INITIAL_ARM_Q,
)

set_arm_q(
    data,
    right_qpos_ids,
    INITIAL_ARM_Q,
)

mujoco.mj_forward(
    model,
    data,
)


left_nominal_rotation = site_rotation(
    model,
    data,
    "left_ee_site",
)

right_nominal_rotation = site_rotation(
    model,
    data,
    "right_ee_site",
)


print(
    "\n"
    + "=" * 100
)

print(
    "3D COMMON-ORIENTATION GOAL-POSE SEARCH"
)

print(
    "=" * 100
)

print(
    "Goal X [m]:",
    GOAL_X,
)

print(
    "Goal reference Z used for IK [m]:",
    GOAL_REFERENCE_Z,
)

print(
    "Position tolerance [mm]:",
    POSITION_TOLERANCE_M * 1000.0,
)

print(
    "Orientation tolerance [deg]:",
    ORIENTATION_TOLERANCE_DEG,
)

print(
    "Clearance ranking safety margin [mm]:",
    CLEARANCE_SAFETY_MARGIN_M * 1000.0,
)

print(
    "Roll values:",
    ROLL_VALUES_DEG,
)

print(
    "Pitch values:",
    PITCH_VALUES_DEG,
)

print(
    "Yaw values:",
    YAW_VALUES_DEG,
)

print(
    "=" * 100
)


# ============================================================
# Solve validated lift
# ============================================================

(
    left_lift_q,
    left_lift_achieved_position,
    left_lift_achieved_rotation,
) = solve_pose_ik(
    model=model,
    data=data,
    joint_names=LEFT_JOINT_NAMES,
    site_name="left_ee_site",
    target_position=LEFT_LIFT_POSITION,
    target_rotation=left_nominal_rotation,
    initial_q=INITIAL_ARM_Q,
)


(
    right_lift_q,
    right_lift_achieved_position,
    right_lift_achieved_rotation,
) = solve_pose_ik(
    model=model,
    data=data,
    joint_names=RIGHT_JOINT_NAMES,
    site_name="right_ee_site",
    target_position=RIGHT_LIFT_POSITION,
    target_rotation=right_nominal_rotation,
    initial_q=INITIAL_ARM_Q,
)


left_lift_position_error = float(
    np.linalg.norm(
        left_lift_achieved_position
        - LEFT_LIFT_POSITION
    )
)

right_lift_position_error = float(
    np.linalg.norm(
        right_lift_achieved_position
        - RIGHT_LIFT_POSITION
    )
)


print(
    "\nValidated lift seed:"
)

print(
    "  Left position error [mm]:",
    left_lift_position_error * 1000.0,
)

print(
    "  Right position error [mm]:",
    right_lift_position_error * 1000.0,
)


# ============================================================
# Solve X=0.40 nominal-orientation seed
# ============================================================

(
    left_seed_q,
    left_seed_achieved_position,
    left_seed_achieved_rotation,
) = solve_pose_ik(
    model=model,
    data=data,
    joint_names=LEFT_JOINT_NAMES,
    site_name="left_ee_site",
    target_position=LEFT_SEED_POSITION,
    target_rotation=left_nominal_rotation,
    initial_q=left_lift_q,
)


(
    right_seed_q,
    right_seed_achieved_position,
    right_seed_achieved_rotation,
) = solve_pose_ik(
    model=model,
    data=data,
    joint_names=RIGHT_JOINT_NAMES,
    site_name="right_ee_site",
    target_position=RIGHT_SEED_POSITION,
    target_rotation=right_nominal_rotation,
    initial_q=right_lift_q,
)


left_seed_position_error = float(
    np.linalg.norm(
        left_seed_achieved_position
        - LEFT_SEED_POSITION
    )
)

right_seed_position_error = float(
    np.linalg.norm(
        right_seed_achieved_position
        - RIGHT_SEED_POSITION
    )
)


left_seed_orientation_error = (
    orientation_error_deg(
        left_nominal_rotation,
        left_seed_achieved_rotation,
    )
)

right_seed_orientation_error = (
    orientation_error_deg(
        right_nominal_rotation,
        right_seed_achieved_rotation,
    )
)


seed_pass = all([
    left_seed_position_error
    <= POSITION_TOLERANCE_M,

    right_seed_position_error
    <= POSITION_TOLERANCE_M,

    left_seed_orientation_error
    <= ORIENTATION_TOLERANCE_DEG,

    right_seed_orientation_error
    <= ORIENTATION_TOLERANCE_DEG,
])


print(
    "\nX=0.40 nominal-orientation seed:"
)

print(
    "  Left position error [mm]:",
    left_seed_position_error * 1000.0,
)

print(
    "  Right position error [mm]:",
    right_seed_position_error * 1000.0,
)

print(
    "  Left orientation error [deg]:",
    left_seed_orientation_error,
)

print(
    "  Right orientation error [deg]:",
    right_seed_orientation_error,
)

print(
    "  Seed PASS:",
    seed_pass,
)


if not seed_pass:

    raise RuntimeError(
        "The X=0.40 seed pose failed. "
        "3D orientation search was not started."
    )


# ============================================================
# Search all common orientations
# ============================================================

successful_candidates = []

total_candidates = (
    len(ROLL_VALUES_DEG)
    * len(PITCH_VALUES_DEG)
    * len(YAW_VALUES_DEG)
)

candidate_index = 0


print(
    "\n"
    + "=" * 100
)

print(
    "SEARCHING COMMON RIGID-BODY ORIENTATIONS"
)

print(
    "=" * 100
)

print(
    "Total candidates:",
    total_candidates,
)


for roll_deg in ROLL_VALUES_DEG:

    for pitch_deg in PITCH_VALUES_DEG:

        for yaw_deg in YAW_VALUES_DEG:

            candidate_index += 1

            common_rotation = (
                common_world_rotation(
                    roll_deg,
                    pitch_deg,
                    yaw_deg,
                )
            )


            left_target_rotation = (
                common_rotation
                @ left_nominal_rotation
            )

            right_target_rotation = (
                common_rotation
                @ right_nominal_rotation
            )


            (
                left_goal_q,
                left_achieved_position,
                left_achieved_rotation,
            ) = solve_pose_ik(
                model=model,
                data=data,
                joint_names=LEFT_JOINT_NAMES,
                site_name="left_ee_site",
                target_position=LEFT_GOAL_POSITION,
                target_rotation=left_target_rotation,
                initial_q=left_seed_q,
            )


            (
                right_goal_q,
                right_achieved_position,
                right_achieved_rotation,
            ) = solve_pose_ik(
                model=model,
                data=data,
                joint_names=RIGHT_JOINT_NAMES,
                site_name="right_ee_site",
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


            maximum_position_error = max(
                left_position_error,
                right_position_error,
            )

            maximum_orientation_error = max(
                left_orientation_error,
                right_orientation_error,
            )


            candidate_pass = all([
                left_position_error
                <= POSITION_TOLERANCE_M,

                right_position_error
                <= POSITION_TOLERANCE_M,

                left_orientation_error
                <= ORIENTATION_TOLERANCE_DEG,

                right_orientation_error
                <= ORIENTATION_TOLERANCE_DEG,
            ])


            if candidate_pass:

                clearance = (
                    geometric_clearance_metrics(
                        common_rotation
                    )
                )


                successful_candidates.append({
                    "roll_deg": float(
                        roll_deg
                    ),

                    "pitch_deg": float(
                        pitch_deg
                    ),

                    "yaw_deg": float(
                        yaw_deg
                    ),

                    "max_position_error_mm": (
                        maximum_position_error
                        * 1000.0
                    ),

                    "max_orientation_error_deg": (
                        maximum_orientation_error
                    ),

                    "half_extent_z": (
                        clearance[
                            "half_extent_z"
                        ]
                    ),

                    "clearance_at_test_height": (
                        clearance[
                            "clearance_at_test_height"
                        ]
                    ),

                    "required_reference_z_for_touch": (
                        clearance[
                            "required_reference_z_for_touch"
                        ]
                    ),

                    "required_reference_z_with_margin": (
                        clearance[
                            "required_reference_z_with_margin"
                        ]
                    ),

                    "left_q": (
                        left_goal_q.copy()
                    ),

                    "right_q": (
                        right_goal_q.copy()
                    ),
                })


            if (
                candidate_index % 100
                == 0
            ):

                print(
                    f"Processed "
                    f"{candidate_index}/"
                    f"{total_candidates} "
                    f"candidates..."
                )


# ============================================================
# Sort successful solutions
# ============================================================

# Primary objective:
# lowest reference height required for collision-free transport.
#
# Secondary objective:
# smallest position IK error.

successful_candidates.sort(
    key=lambda candidate: (
        candidate[
            "required_reference_z_with_margin"
        ],
        candidate[
            "max_position_error_mm"
        ],
    )
)


# ============================================================
# Summary
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "SEARCH SUMMARY"
)

print(
    "=" * 100
)

print(
    "Total candidates tested:",
    total_candidates,
)

print(
    "Successful goal-pose IK candidates:",
    len(successful_candidates),
)


if not successful_candidates:

    print(
        "\nNo common roll/pitch/yaw orientation "
        "satisfied the goal-pose IK tolerances."
    )

    raise SystemExit(0)


print(
    "\nTop candidates ranked by required "
    "reference Z for 2 mm table clearance:"
)

print(
    "-" * 100
)


for rank, candidate in enumerate(
    successful_candidates[
        :TOP_CANDIDATE_COUNT
    ],
    start=1,
):

    print(
        f"{rank:2d} | "
        f"roll={candidate['roll_deg']:+5.1f} deg | "
        f"pitch={candidate['pitch_deg']:+5.1f} deg | "
        f"yaw={candidate['yaw_deg']:+5.1f} deg | "
        f"pos_err={candidate['max_position_error_mm']:.3f} mm | "
        f"ori_err={candidate['max_orientation_error_deg']:.4f} deg | "
        f"halfZ={candidate['half_extent_z']:.5f} m | "
        f"clear@0.58="
        f"{candidate['clearance_at_test_height'] * 1000.0:+.2f} mm | "
        f"requiredZ+2mm="
        f"{candidate['required_reference_z_with_margin']:.5f} m"
    )


# ============================================================
# Best candidate details
# ============================================================

best = successful_candidates[0]


print(
    "\n"
    + "=" * 100
)

print(
    "BEST GEOMETRIC CANDIDATE"
)

print(
    "=" * 100
)

print(
    "Roll [deg]:",
    best["roll_deg"],
)

print(
    "Pitch [deg]:",
    best["pitch_deg"],
)

print(
    "Yaw [deg]:",
    best["yaw_deg"],
)

print(
    "Maximum position error [mm]:",
    best["max_position_error_mm"],
)

print(
    "Maximum orientation error [deg]:",
    best["max_orientation_error_deg"],
)

print(
    "Object vertical half-extent [m]:",
    best["half_extent_z"],
)

print(
    "Estimated table clearance at "
    "reference Z=0.58 [mm]:",
    best[
        "clearance_at_test_height"
    ] * 1000.0,
)

print(
    "Reference Z required for touch [m]:",
    best[
        "required_reference_z_for_touch"
    ],
)

print(
    "Reference Z required for "
    "2 mm clearance [m]:",
    best[
        "required_reference_z_with_margin"
    ],
)

print(
    "\nLeft goal q:"
)

print(
    best["left_q"]
)

print(
    "\nRight goal q:"
)

print(
    best["right_q"]
)

print(
    "=" * 100
)
