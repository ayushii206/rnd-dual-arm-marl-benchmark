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

TABLE_TOP_Z = 0.50
OBJECT_HALF_Z = 0.06

FLAT_PLACEMENT_Z = (
    TABLE_TOP_Z
    + OBJECT_HALF_Z
)

POSITION_TOLERANCE_M = 1.0e-3
ORIENTATION_TOLERANCE_DEG = 1.0


# Test the part of the goal region that can contain the
# full 0.40 m-long object.
#
# x = 0.430 is the theoretical minimum for complete
# containment in the current goal region.

PLACEMENT_X_VALUES = np.arange(
    0.425,
    0.451,
    0.001,
)


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


# x=0.40 with the nominal orientation is already known
# to be reachable and gives us a good continuation seed.

SEED_X = 0.40

LEFT_SEED_POSITION = np.array([
    SEED_X,
    GRASP_Y,
    FLAT_PLACEMENT_Z,
])

RIGHT_SEED_POSITION = np.array([
    SEED_X,
    -GRASP_Y,
    FLAT_PLACEMENT_Z,
])


# ============================================================
# Helpers
# ============================================================

def get_qpos_ids(
    model,
    joint_names,
):
    ids = []

    for joint_name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
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
    data.qpos[
        qpos_ids
    ] = q


def get_site_rotation(
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
# Capture nominal gripper orientations
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
# Solve original lift
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
# Solve flat x=0.40 seed
# ============================================================

(
    left_previous_q,
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
    right_previous_q,
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
    "\n"
    + "=" * 90
)

print(
    "FLAT GOAL-PLACEMENT REACHABILITY"
)

print(
    "=" * 90
)

print(
    "Placement Z [m]:",
    FLAT_PLACEMENT_Z,
)

print(
    "Goal full-containment minimum X [m]:",
    0.430,
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
    "\nFlat x=0.40 seed:"
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
    "  PASS:",
    seed_pass,
)


if not seed_pass:

    raise RuntimeError(
        "Flat x=0.40 placement seed failed."
    )


# ============================================================
# Sequential X scan
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "SCANNING FLAT PLACEMENT X"
)

print(
    "=" * 90
)


furthest_pass_x = None

first_fail_x = None


for placement_x in PLACEMENT_X_VALUES:

    left_target_position = np.array([
        placement_x,
        GRASP_Y,
        FLAT_PLACEMENT_Z,
    ])

    right_target_position = np.array([
        placement_x,
        -GRASP_Y,
        FLAT_PLACEMENT_Z,
    ])


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
        initial_q=left_previous_q,
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
        target_rotation=right_nominal_rotation,
        initial_q=right_previous_q,
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


    placement_pass = all([
        left_position_error
        <= POSITION_TOLERANCE_M,

        right_position_error
        <= POSITION_TOLERANCE_M,

        left_orientation_error
        <= ORIENTATION_TOLERANCE_DEG,

        right_orientation_error
        <= ORIENTATION_TOLERANCE_DEG,
    ])


    inside_goal = (
        placement_x
        >= 0.430
    )


    print(
        f"x={placement_x:.3f} m "
        f"| Lpos="
        f"{left_position_error * 1000.0:.3f} mm "
        f"| Rpos="
        f"{right_position_error * 1000.0:.3f} mm "
        f"| Lori="
        f"{left_orientation_error:.4f} deg "
        f"| Rori="
        f"{right_orientation_error:.4f} deg "
        f"| goal={inside_goal} "
        f"| PASS={placement_pass}"
    )


    if placement_pass:

        furthest_pass_x = float(
            placement_x
        )

        left_previous_q = (
            left_q.copy()
        )

        right_previous_q = (
            right_q.copy()
        )

    else:

        first_fail_x = float(
            placement_x
        )

        break


# ============================================================
# Result
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "FLAT PLACEMENT RESULT"
)

print(
    "=" * 90
)

print(
    "Furthest reachable flat X [m]:",
    furthest_pass_x,
)

print(
    "First failed flat X [m]:",
    first_fail_x,
)


if (
    furthest_pass_x is not None
    and furthest_pass_x >= 0.430
):

    print(
        "\nRESULT: A flat placement pose "
        "inside the goal region is kinematically reachable."
    )

else:

    print(
        "\nRESULT: The tested flat placement poses "
        "did not reach the full-containment region."
    )


print(
    "=" * 90
)
