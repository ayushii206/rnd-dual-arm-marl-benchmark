import numpy as np
import mujoco

from src.controllers.ik_controller import solve_pose_ik


# ============================================================
# Model
# ============================================================

MODEL_PATH = (
    "assets/models/dual_panda/transport_scene.xml"
)


# ============================================================
# Task geometry
# ============================================================

TABLE_TOP_Z = 0.50

GOAL_CENTER_X = 0.45
GOAL_HALF_X = 0.22

GOAL_LEFT_X = (
    GOAL_CENTER_X
    - GOAL_HALF_X
)

OBJECT_HALF_X = 0.20
OBJECT_HALF_Y = 0.12
OBJECT_HALF_Z = 0.06


# ============================================================
# Validated grasp geometry
# ============================================================

GRASP_Y = 0.145

# Bilateral EE reference is the midpoint between both grippers.

LEFT_GRASP_OFFSET = np.array([
    0.0,
    GRASP_Y,
    0.0,
])

RIGHT_GRASP_OFFSET = np.array([
    0.0,
    -GRASP_Y,
    0.0,
])


# Observed during stable lift:
#
# EE midpoint reference Z = 0.580000
# object center Z          ≈ 0.573985

OBJECT_CENTER_OFFSET = np.array([
    0.0,
    0.0,
    -0.006015,
])


# ============================================================
# IK tolerances
# ============================================================

POSITION_TOLERANCE_M = 1.0e-3
ORIENTATION_TOLERANCE_DEG = 1.0


# ============================================================
# Placement margin
# ============================================================

# First use zero extra margin.
#
# A candidate must geometrically fit completely inside the goal,
# but we do not yet demand another few mm of buffer.

GOAL_CONTAINMENT_MARGIN_M = 0.0

# Placement pose is computed with the lowest point of the
# object exactly at table height.

TABLE_CLEARANCE_MARGIN_M = 0.0


# ============================================================
# Orientation search
# ============================================================

ROLL_VALUES_DEG = np.arange(
    -8.0,
    8.01,
    1.0,
)

PITCH_VALUES_DEG = np.arange(
    -12.0,
    4.01,
    1.0,
)

YAW_VALUES_DEG = np.arange(
    -8.0,
    8.01,
    1.0,
)


# ============================================================
# Robot
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


# Known reachable configuration used only as an IK seed.

SEED_REFERENCE_POSITION = np.array([
    0.40,
    0.0,
    0.58,
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
        .reshape(3, 3)
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
        np.trace(rotation_error)
        - 1.0
    ) / 2.0

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(cos_angle)
        )
    )


def rotation_x_world(
    angle_deg,
):
    angle = np.radians(
        angle_deg
    )

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def rotation_y_world(
    angle_deg,
):
    angle = np.radians(
        angle_deg
    )

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rotation_z_world(
    angle_deg,
):
    angle = np.radians(
        angle_deg
    )

    c = np.cos(angle)
    s = np.sin(angle)

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
    return (
        rotation_z_world(yaw_deg)
        @ rotation_y_world(pitch_deg)
        @ rotation_x_world(roll_deg)
    )


def oriented_box_half_extent_x(
    rotation,
):
    return float(
        abs(rotation[0, 0])
        * OBJECT_HALF_X
        +
        abs(rotation[0, 1])
        * OBJECT_HALF_Y
        +
        abs(rotation[0, 2])
        * OBJECT_HALF_Z
    )


def oriented_box_half_extent_z(
    rotation,
):
    return float(
        abs(rotation[2, 0])
        * OBJECT_HALF_X
        +
        abs(rotation[2, 1])
        * OBJECT_HALF_Y
        +
        abs(rotation[2, 2])
        * OBJECT_HALF_Z
    )


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
# Capture nominal EE rotations
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


# ============================================================
# Solve validated lift
# ============================================================

left_lift_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=LEFT_JOINT_NAMES,
    site_name="left_ee_site",
    target_position=LEFT_LIFT_POSITION,
    target_rotation=left_nominal_rotation,
    initial_q=INITIAL_ARM_Q,
)


right_lift_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=RIGHT_JOINT_NAMES,
    site_name="right_ee_site",
    target_position=RIGHT_LIFT_POSITION,
    target_rotation=right_nominal_rotation,
    initial_q=INITIAL_ARM_Q,
)


# ============================================================
# Solve known x=0.40 nominal seed
# ============================================================

left_seed_position = (
    SEED_REFERENCE_POSITION
    + LEFT_GRASP_OFFSET
)

right_seed_position = (
    SEED_REFERENCE_POSITION
    + RIGHT_GRASP_OFFSET
)


left_seed_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=LEFT_JOINT_NAMES,
    site_name="left_ee_site",
    target_position=left_seed_position,
    target_rotation=left_nominal_rotation,
    initial_q=left_lift_q,
)


right_seed_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=RIGHT_JOINT_NAMES,
    site_name="right_ee_site",
    target_position=right_seed_position,
    target_rotation=right_nominal_rotation,
    initial_q=right_lift_q,
)


# ============================================================
# Search
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
    + "=" * 110
)

print(
    "GOAL PLACEMENT POSE FEASIBILITY GRID"
)

print(
    "=" * 110
)

print(
    "Goal X bounds [m]:",
    (
        GOAL_CENTER_X - GOAL_HALF_X,
        GOAL_CENTER_X + GOAL_HALF_X,
    ),
)

print(
    "Table top Z [m]:",
    TABLE_TOP_Z,
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
    "Total orientation candidates:",
    total_candidates,
)

print(
    "=" * 110
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


            # ------------------------------------------------
            # Object projected size
            # ------------------------------------------------

            half_extent_x = (
                oriented_box_half_extent_x(
                    common_rotation
                )
            )

            half_extent_z = (
                oriented_box_half_extent_z(
                    common_rotation
                )
            )


            # ------------------------------------------------
            # Minimum object center X for complete goal
            # containment.
            # ------------------------------------------------

            object_center_x = (
                GOAL_LEFT_X
                + half_extent_x
                + GOAL_CONTAINMENT_MARGIN_M
            )


            # ------------------------------------------------
            # Determine reference Z so the lowest point of
            # the object lies exactly on/above the table.
            # ------------------------------------------------

            rotated_object_center_offset = (
                common_rotation
                @ OBJECT_CENTER_OFFSET
            )

            object_center_offset_z = float(
                rotated_object_center_offset[2]
            )


            reference_z = (
                TABLE_TOP_Z
                + TABLE_CLEARANCE_MARGIN_M
                + half_extent_z
                - object_center_offset_z
            )


            reference_position = np.array([
                object_center_x,
                0.0,
                reference_z,
            ])


            # ------------------------------------------------
            # Rotate the grasp points with the object.
            #
            # This is important:
            # roll/yaw change EE positions as well as
            # orientations.
            # ------------------------------------------------

            left_target_position = (
                reference_position
                + common_rotation
                @ LEFT_GRASP_OFFSET
            )

            right_target_position = (
                reference_position
                + common_rotation
                @ RIGHT_GRASP_OFFSET
            )


            left_target_rotation = (
                common_rotation
                @ left_nominal_rotation
            )

            right_target_rotation = (
                common_rotation
                @ right_nominal_rotation
            )


            # ------------------------------------------------
            # IK
            # ------------------------------------------------

            (
                left_q,
                left_achieved_position,
                left_achieved_rotation,
            ) = solve_pose_ik(
                model=model,
                data=data,
                joint_names=LEFT_JOINT_NAMES,
                site_name="left_ee_site",
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
                joint_names=RIGHT_JOINT_NAMES,
                site_name="right_ee_site",
                target_position=right_target_position,
                target_rotation=right_target_rotation,
                initial_q=right_seed_q,
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

                    "object_center_x": float(
                        object_center_x
                    ),

                    "reference_z": float(
                        reference_z
                    ),

                    "half_extent_x": float(
                        half_extent_x
                    ),

                    "half_extent_z": float(
                        half_extent_z
                    ),

                    "max_position_error_mm": float(
                        max_position_error
                        * 1000.0
                    ),

                    "max_orientation_error_deg": float(
                        max_orientation_error
                    ),

                    "left_target_position": (
                        left_target_position.copy()
                    ),

                    "right_target_position": (
                        right_target_position.copy()
                    ),

                    "left_q": (
                        left_q.copy()
                    ),

                    "right_q": (
                        right_q.copy()
                    ),
                })


            if candidate_index % 500 == 0:

                print(
                    f"Processed "
                    f"{candidate_index}/"
                    f"{total_candidates}..."
                )


# ============================================================
# Sort
# ============================================================

# Prefer:
#
# 1. smallest object center X
# 2. smallest required reference Z
# 3. smallest IK position error

successful_candidates.sort(
    key=lambda candidate: (
        candidate[
            "object_center_x"
        ],
        candidate[
            "reference_z"
        ],
        candidate[
            "max_position_error_mm"
        ],
    )
)


# ============================================================
# Results
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "SEARCH RESULT"
)

print(
    "=" * 110
)

print(
    "Successful physically valid placement poses:",
    len(successful_candidates),
)


if not successful_candidates:

    print(
        "\nNo candidate simultaneously satisfied:"
    )

    print(
        "  - full X containment in the goal,"
    )

    print(
        "  - table-contact geometry,"
    )

    print(
        "  - bilateral rigid grasp,"
    )

    print(
        "  - 1 mm position IK tolerance,"
    )

    print(
        "  - 1 degree orientation IK tolerance."
    )

    print(
        "=" * 110
    )

    raise SystemExit(0)


print(
    "\nTop 30 candidates:"
)

print(
    "-" * 110
)


for rank, candidate in enumerate(
    successful_candidates[:30],
    start=1,
):

    print(
        f"{rank:2d} | "
        f"R={candidate['roll_deg']:+5.1f} "
        f"P={candidate['pitch_deg']:+5.1f} "
        f"Y={candidate['yaw_deg']:+5.1f} deg | "
        f"obj_x={candidate['object_center_x']:.5f} | "
        f"ref_z={candidate['reference_z']:.5f} | "
        f"halfX={candidate['half_extent_x']:.5f} | "
        f"halfZ={candidate['half_extent_z']:.5f} | "
        f"pos={candidate['max_position_error_mm']:.3f} mm | "
        f"ori={candidate['max_orientation_error_deg']:.4f} deg"
    )


best = successful_candidates[0]


print(
    "\n"
    + "=" * 110
)

print(
    "BEST PLACEMENT CANDIDATE"
)

print(
    "=" * 110
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
    "Required object center X [m]:",
    best["object_center_x"],
)

print(
    "Required EE reference Z [m]:",
    best["reference_z"],
)

print(
    "Object projected X half-extent [m]:",
    best["half_extent_x"],
)

print(
    "Object projected Z half-extent [m]:",
    best["half_extent_z"],
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
    "Left EE target position:"
)

print(
    best["left_target_position"]
)

print(
    "Right EE target position:"
)

print(
    best["right_target_position"]
)

print(
    "Left q:"
)

print(
    best["left_q"]
)

print(
    "Right q:"
)

print(
    best["right_q"]
)

print(
    "=" * 110
)
