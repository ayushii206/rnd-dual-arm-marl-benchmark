import numpy as np
import mujoco

from src.controllers.ik_controller import solve_pose_ik


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = (
    "assets/models/dual_panda/transport_scene.xml"
)

GRASP_Y = 0.145

POSITION_TOLERANCE_M = 1.0e-3
ORIENTATION_TOLERANCE_DEG = 1.0


# ============================================================
# Placement region to investigate
# ============================================================

# We already know that the cooperative system is comfortable
# around x = 0.40 m.
#
# We now scan a small region around that point to determine
# the furthest flat, near-table placement configuration that
# both arms can satisfy while preserving the nominal rigid
# grasp orientation.

PLACEMENT_X_VALUES = np.arange(
    0.390,
    0.411,
    0.001,
)


# ============================================================
# Placement height
# ============================================================

# From the stable cooperative lift:
#
# EE reference midpoint Z = 0.580000 m
# settled object center Z  ≈ 0.573985 m
#
# Therefore:
#
# object center - EE reference
# ≈ -0.006015 m
#
# For the box to sit flat on the table:
#
# table top Z       = 0.500 m
# object half-height = 0.060 m
#
# desired object center Z = 0.560 m
#
# Therefore the expected EE reference height is approximately:
#
# 0.560 + 0.006015 = 0.566015 m
#
# We scan around this value.

PLACEMENT_Z_VALUES = np.arange(
    0.563,
    0.571,
    0.001,
)


TABLE_TOP_Z = 0.50

OBJECT_HALF_Z = 0.06

OBJECT_CENTER_OFFSET_Z = -0.006015


# ============================================================
# Robot configuration
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


# Original validated lifted grasp.

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


# ============================================================
# Helper functions
# ============================================================

def get_qpos_ids(
    model,
    joint_names,
):
    """
    Return the qpos indices corresponding to the given joints.
    """

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

    return ids


def set_arm_q(
    data,
    qpos_ids,
    q,
):
    """
    Set one arm's joint configuration directly in qpos.
    """

    data.qpos[
        qpos_ids
    ] = q


def get_site_rotation(
    model,
    data,
    site_name,
):
    """
    Return a site's world-frame 3x3 rotation matrix.
    """

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id < 0:
        raise RuntimeError(
            f"Site not found: {site_name}"
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
    """
    Compute angular difference between two rotation matrices.
    """

    rotation_error = (
        target_rotation.T
        @ achieved_rotation
    )

    cos_angle = (
        np.trace(
            rotation_error
        )
        - 1.0
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


# ============================================================
# Load MuJoCo model
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
# Capture nominal end-effector orientations
# ============================================================

# Put both arms at the known nominal configuration first.

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


left_nominal_rotation = (
    get_site_rotation(
        model,
        data,
        "left_ee_site",
    )
)

right_nominal_rotation = (
    get_site_rotation(
        model,
        data,
        "right_ee_site",
    )
)


# ============================================================
# Solve the validated lift configuration
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


left_lift_orientation_error = (
    orientation_error_deg(
        left_nominal_rotation,
        left_lift_achieved_rotation,
    )
)

right_lift_orientation_error = (
    orientation_error_deg(
        right_nominal_rotation,
        right_lift_achieved_rotation,
    )
)


lift_pass = all([
    left_lift_position_error
    <= POSITION_TOLERANCE_M,

    right_lift_position_error
    <= POSITION_TOLERANCE_M,

    left_lift_orientation_error
    <= ORIENTATION_TOLERANCE_DEG,

    right_lift_orientation_error
    <= ORIENTATION_TOLERANCE_DEG,
])


print(
    "\n"
    + "=" * 100
)

print(
    "REACHABLE FLAT PLACEMENT REGION"
)

print(
    "=" * 100
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
    "X scan range [m]:",
    (
        PLACEMENT_X_VALUES[0],
        PLACEMENT_X_VALUES[-1],
    ),
)

print(
    "EE reference Z scan range [m]:",
    (
        PLACEMENT_Z_VALUES[0],
        PLACEMENT_Z_VALUES[-1],
    ),
)

print(
    "Estimated nominal flat-placement EE Z [m]:",
    0.560 - OBJECT_CENTER_OFFSET_Z,
)

print(
    "\nValidated lift:"
)

print(
    "  Left position error [mm]:",
    left_lift_position_error * 1000.0,
)

print(
    "  Right position error [mm]:",
    right_lift_position_error * 1000.0,
)

print(
    "  Left orientation error [deg]:",
    left_lift_orientation_error,
)

print(
    "  Right orientation error [deg]:",
    right_lift_orientation_error,
)

print(
    "  PASS:",
    lift_pass,
)

print(
    "=" * 100
)


if not lift_pass:
    raise RuntimeError(
        "Validated lift configuration failed. "
        "Do not continue placement scan."
    )


# ============================================================
# Scan X / Z placement region
# ============================================================

successful_candidates = []


for placement_x in PLACEMENT_X_VALUES:

    print(
        "\n"
        + "-" * 100
    )

    print(
        f"X = {placement_x:.3f} m"
    )

    print(
        "-" * 100
    )


    for placement_z in PLACEMENT_Z_VALUES:

        left_target_position = np.array([
            placement_x,
            GRASP_Y,
            placement_z,
        ])

        right_target_position = np.array([
            placement_x,
            -GRASP_Y,
            placement_z,
        ])


        # ----------------------------------------------------
        # Solve left arm
        # ----------------------------------------------------

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
            target_rotation=left_nominal_rotation,
            initial_q=left_lift_q,
        )


        # ----------------------------------------------------
        # Solve right arm
        # ----------------------------------------------------

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
            target_rotation=right_nominal_rotation,
            initial_q=right_lift_q,
        )


        # ----------------------------------------------------
        # Position errors
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Orientation errors
        # ----------------------------------------------------

        left_orientation_error = (
            orientation_error_deg(
                left_nominal_rotation,
                left_achieved_rotation,
            )
        )

        right_orientation_error = (
            orientation_error_deg(
                right_nominal_rotation,
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


        # ----------------------------------------------------
        # IK acceptance
        # ----------------------------------------------------

        placement_pass = all([
            max_position_error
            <= POSITION_TOLERANCE_M,

            max_orientation_error
            <= ORIENTATION_TOLERANCE_DEG,
        ])


        # ----------------------------------------------------
        # Estimate physical table clearance
        # ----------------------------------------------------

        estimated_object_center_z = (
            placement_z
            + OBJECT_CENTER_OFFSET_Z
        )

        estimated_object_bottom_z = (
            estimated_object_center_z
            - OBJECT_HALF_Z
        )

        estimated_table_gap_m = (
            estimated_object_bottom_z
            - TABLE_TOP_Z
        )

        estimated_table_gap_mm = (
            estimated_table_gap_m
            * 1000.0
        )


        # ----------------------------------------------------
        # Print candidate
        # ----------------------------------------------------

        print(
            f"  Z={placement_z:.3f} m "
            f"| pos={max_position_error * 1000.0:.3f} mm "
            f"| ori={max_orientation_error:.4f} deg "
            f"| table_gap={estimated_table_gap_mm:+.2f} mm "
            f"| PASS={placement_pass}"
        )


        # ----------------------------------------------------
        # Save successful candidate
        # ----------------------------------------------------

        if placement_pass:

            successful_candidates.append({
                "x": float(
                    placement_x
                ),

                "z": float(
                    placement_z
                ),

                "position_error_mm": float(
                    max_position_error
                    * 1000.0
                ),

                "orientation_error_deg": float(
                    max_orientation_error
                ),

                "table_gap_mm": float(
                    estimated_table_gap_mm
                ),

                "left_q": (
                    left_q.copy()
                ),

                "right_q": (
                    right_q.copy()
                ),
            })


# ============================================================
# Identify physically meaningful candidates
# ============================================================

# The -6.015 mm object/reference offset was measured from
# simulation and is therefore approximate.
#
# We accept candidates whose estimated object bottom lies
# within +/- 1 mm of the table surface.

TABLE_GAP_TOLERANCE_MM = 1.0


near_table_candidates = [
    candidate
    for candidate in successful_candidates
    if abs(
        candidate[
            "table_gap_mm"
        ]
    )
    <= TABLE_GAP_TOLERANCE_MM
]


# ============================================================
# Summary
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "SUMMARY"
)

print(
    "=" * 100
)

print(
    "Total X/Z combinations tested:",
    (
        len(PLACEMENT_X_VALUES)
        * len(PLACEMENT_Z_VALUES)
    ),
)

print(
    "Total IK-successful candidates:",
    len(
        successful_candidates
    ),
)

print(
    "IK-successful candidates within "
    "+/- 1 mm estimated table gap:",
    len(
        near_table_candidates
    ),
)


# ============================================================
# Print near-table candidates
# ============================================================

if near_table_candidates:

    # Prefer:
    #
    # 1. furthest X
    # 2. smallest absolute table gap
    # 3. smallest IK error

    near_table_candidates.sort(
        key=lambda candidate: (
            -candidate[
                "x"
            ],

            abs(
                candidate[
                    "table_gap_mm"
                ]
            ),

            candidate[
                "position_error_mm"
            ],
        )
    )


    print(
        "\nBest near-table candidates:"
    )

    print(
        "-" * 100
    )


    for candidate in (
        near_table_candidates[:20]
    ):

        print(
            f"x={candidate['x']:.3f} m "
            f"| Z={candidate['z']:.3f} m "
            f"| pos="
            f"{candidate['position_error_mm']:.3f} mm "
            f"| ori="
            f"{candidate['orientation_error_deg']:.4f} deg "
            f"| gap="
            f"{candidate['table_gap_mm']:+.2f} mm"
        )


    # ========================================================
    # Best candidate
    # ========================================================

    best = (
        near_table_candidates[0]
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "FURTHEST REACHABLE NEAR-TABLE FLAT POSE"
    )

    print(
        "=" * 100
    )

    print(
        "X [m]:",
        best["x"],
    )

    print(
        "EE reference Z [m]:",
        best["z"],
    )

    print(
        "Maximum position error [mm]:",
        best[
            "position_error_mm"
        ],
    )

    print(
        "Maximum orientation error [deg]:",
        best[
            "orientation_error_deg"
        ],
    )

    print(
        "Estimated table gap [mm]:",
        best[
            "table_gap_mm"
        ],
    )


    print(
        "\nLeft q:"
    )

    print(
        best[
            "left_q"
        ]
    )


    print(
        "\nRight q:"
    )

    print(
        best[
            "right_q"
        ]
    )


else:

    print(
        "\nNo IK-valid flat placement pose "
        "was found within +/- 1 mm of "
        "the estimated table surface."
    )


print(
    "=" * 100
)
