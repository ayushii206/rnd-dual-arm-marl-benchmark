import numpy as np
import mujoco

from src.controllers.ik_controller import solve_pose_ik


MODEL_PATH = (
    "assets/models/dual_panda/transport_scene.xml"
)


# ============================================================
# Configuration
# ============================================================

GOAL_TRANSPORT_DISTANCE_X = 0.45

GOAL_TRAJECTORY_WAYPOINTS = 46

POSITION_TOLERANCE_M = 1.0e-3
ORIENTATION_TOLERANCE_DEG = 1.0


# Test the narrow region we actually care about.
TRANSPORT_Z_VALUES = np.arange(
    0.580,
    0.596,
    0.001,
)

COMMON_Y_ANGLE_VALUES_DEG = [
    -8.7,
    -9.0,
    -9.3,
    -9.5,
    -9.7,
    -10.0,
]


INITIAL_ARM_Q = np.array([
    0.0,
    -0.4,
    0.0,
    -1.8,
    0.0,
    1.4,
    0.8,
])


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


LEFT_JOINT_NAMES = [
    f"left_joint{i}"
    for i in range(1, 8)
]

RIGHT_JOINT_NAMES = [
    f"right_joint{i}"
    for i in range(1, 8)
]


# ============================================================
# Helpers
# ============================================================

def smoothstep(alpha):
    return (
        alpha
        * alpha
        * (
            3.0
            - 2.0 * alpha
        )
    )


def rotation_y_world(angle_deg):
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


def orientation_error_deg(
    target_rotation,
    achieved_rotation,
):
    rotation_error = (
        target_rotation.T
        @ achieved_rotation
    )

    trace_value = np.trace(
        rotation_error
    )

    cos_angle = (
        trace_value - 1.0
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


def set_qpos(
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

set_qpos(
    data,
    left_qpos_ids,
    INITIAL_ARM_Q,
)

set_qpos(
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
# Solve validated 0.58 m lift
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
# Scan one Z/angle combination
# ============================================================

def test_combination(
    transport_z,
    final_angle_deg,
):

    left_start_position = np.array([
        0.0,
        0.145,
        transport_z,
    ])

    right_start_position = np.array([
        0.0,
        -0.145,
        transport_z,
    ])


    # --------------------------------------------------------
    # Solve clearance configuration first
    # --------------------------------------------------------

    (
        left_start_q,
        left_start_achieved_position,
        left_start_achieved_rotation,
    ) = solve_pose_ik(
        model=model,
        data=data,
        joint_names=LEFT_JOINT_NAMES,
        site_name="left_ee_site",
        target_position=left_start_position,
        target_rotation=left_nominal_rotation,
        initial_q=left_lift_q,
    )


    (
        right_start_q,
        right_start_achieved_position,
        right_start_achieved_rotation,
    ) = solve_pose_ik(
        model=model,
        data=data,
        joint_names=RIGHT_JOINT_NAMES,
        site_name="right_ee_site",
        target_position=right_start_position,
        target_rotation=right_nominal_rotation,
        initial_q=right_lift_q,
    )


    left_start_position_error = float(
        np.linalg.norm(
            left_start_achieved_position
            - left_start_position
        )
    )

    right_start_position_error = float(
        np.linalg.norm(
            right_start_achieved_position
            - right_start_position
        )
    )


    left_start_orientation_error = (
        orientation_error_deg(
            left_nominal_rotation,
            left_start_achieved_rotation,
        )
    )

    right_start_orientation_error = (
        orientation_error_deg(
            right_nominal_rotation,
            right_start_achieved_rotation,
        )
    )


    start_pass = all([
        left_start_position_error
        <= POSITION_TOLERANCE_M,

        right_start_position_error
        <= POSITION_TOLERANCE_M,

        left_start_orientation_error
        <= ORIENTATION_TOLERANCE_DEG,

        right_start_orientation_error
        <= ORIENTATION_TOLERANCE_DEG,
    ])


    if not start_pass:
        return {
            "pass": False,
            "failed_waypoint": 0,
            "failed_x": 0.0,
            "max_position_error_mm": (
                max(
                    left_start_position_error,
                    right_start_position_error,
                )
                * 1000.0
            ),
        }


    reference_position = np.array([
        0.0,
        0.0,
        transport_z,
    ])


    left_grasp_offset = np.array([
        0.0,
        0.145,
        0.0,
    ])

    right_grasp_offset = np.array([
        0.0,
        -0.145,
        0.0,
    ])


    previous_left_q = (
        left_start_q.copy()
    )

    previous_right_q = (
        right_start_q.copy()
    )


    maximum_position_error = 0.0


    # --------------------------------------------------------
    # Sequential continuation IK
    # --------------------------------------------------------

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
            final_angle_deg
            * trajectory_alpha
        )


        common_rotation = rotation_y_world(
            current_angle_deg
        )


        current_reference_position = (
            reference_position
            + np.array([
                current_x,
                0.0,
                0.0,
            ])
        )


        left_target_position = (
            current_reference_position
            + common_rotation
            @ left_grasp_offset
        )


        right_target_position = (
            current_reference_position
            + common_rotation
            @ right_grasp_offset
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
            initial_q=previous_left_q,
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


        maximum_position_error = max(
            maximum_position_error,
            left_position_error,
            right_position_error,
        )


        waypoint_pass = all([
            left_position_error
            <= POSITION_TOLERANCE_M,

            right_position_error
            <= POSITION_TOLERANCE_M,

            left_orientation_error
            <= ORIENTATION_TOLERANCE_DEG,

            right_orientation_error
            <= ORIENTATION_TOLERANCE_DEG,
        ])


        if not waypoint_pass:

            return {
                "pass": False,
                "failed_waypoint": (
                    waypoint_index
                ),
                "failed_x": (
                    current_x
                ),
                "max_position_error_mm": (
                    maximum_position_error
                    * 1000.0
                ),
            }


        previous_left_q = (
            left_q.copy()
        )

        previous_right_q = (
            right_q.copy()
        )


    return {
        "pass": True,
        "failed_waypoint": None,
        "failed_x": (
            GOAL_TRANSPORT_DISTANCE_X
        ),
        "max_position_error_mm": (
            maximum_position_error
            * 1000.0
        ),
    }


# ============================================================
# Run grid
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "GOAL HEIGHT / ORIENTATION FEASIBILITY GRID"
)

print(
    "=" * 90
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
    "=" * 90
)


successful_combinations = []


for transport_z in TRANSPORT_Z_VALUES:

    print(
        f"\nZ = {transport_z:.3f} m"
    )

    for final_angle_deg in (
        COMMON_Y_ANGLE_VALUES_DEG
    ):

        result = test_combination(
            transport_z,
            final_angle_deg,
        )


        if result["pass"]:

            successful_combinations.append(
                (
                    transport_z,
                    final_angle_deg,
                    result[
                        "max_position_error_mm"
                    ],
                )
            )

            print(
                f"  angle={final_angle_deg:+5.1f} deg "
                f"| PASS "
                f"| max pos err="
                f"{result['max_position_error_mm']:.3f} mm"
            )

        else:

            print(
                f"  angle={final_angle_deg:+5.1f} deg "
                f"| FAIL "
                f"| wp={result['failed_waypoint']} "
                f"| x={result['failed_x']:.4f} m "
                f"| max pos err="
                f"{result['max_position_error_mm']:.3f} mm"
            )


print(
    "\n"
    + "=" * 90
)

print(
    "SUCCESSFUL FULL-TRAJECTORY COMBINATIONS"
)

print(
    "=" * 90
)


if successful_combinations:

    for (
        transport_z,
        angle_deg,
        max_error_mm,
    ) in successful_combinations:

        print(
            f"Z={transport_z:.3f} m "
            f"| angle={angle_deg:+.1f} deg "
            f"| max error={max_error_mm:.3f} mm"
        )

else:

    print(
        "No tested combination completed "
        "the full trajectory."
    )


print(
    "=" * 90
)
