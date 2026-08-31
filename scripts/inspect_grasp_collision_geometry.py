from pathlib import Path

import mujoco
import numpy as np

from src.controllers.ik_controller import solve_pose_ik


MODEL_PATH = Path(
    "assets/models/dual_panda/transport_scene.xml"
)


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)


# ------------------------------------------------------------
# Joint names
# ------------------------------------------------------------

left_joint_names = [
    f"left_joint{i}"
    for i in range(1, 8)
]

right_joint_names = [
    f"right_joint{i}"
    for i in range(1, 8)
]


# ------------------------------------------------------------
# Initial arm configuration
# ------------------------------------------------------------

initial_q = np.array([
    0.0,
    -0.4,
    0.0,
    -1.8,
    0.0,
    1.4,
    0.8,
])


# ------------------------------------------------------------
# Helper: set arm qpos
# ------------------------------------------------------------

def set_arm_qpos(
    joint_names,
    q,
):

    for joint_name, value in zip(
        joint_names,
        q,
    ):

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_id = model.jnt_qposadr[
            joint_id
        ]

        data.qpos[
            qpos_id
        ] = value


# ------------------------------------------------------------
# Set both arms to nominal pose
# ------------------------------------------------------------

set_arm_qpos(
    left_joint_names,
    initial_q,
)

set_arm_qpos(
    right_joint_names,
    initial_q,
)


# ------------------------------------------------------------
# Open all four fingers directly
# ------------------------------------------------------------

for prefix in [
    "left",
    "right",
]:

    for finger_index in [
        1,
        2,
    ]:

        joint_name = (
            f"{prefix}_finger_joint"
            f"{finger_index}"
        )

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_id = model.jnt_qposadr[
            joint_id
        ]

        data.qpos[
            qpos_id
        ] = 0.04


mujoco.mj_forward(
    model,
    data,
)


# ------------------------------------------------------------
# Get nominal EE orientations
# ------------------------------------------------------------

left_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "left_ee_site",
)

right_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_ee_site",
)


left_rotation = (
    data.site_xmat[
        left_site_id
    ]
    .reshape(3, 3)
    .copy()
)

right_rotation = (
    data.site_xmat[
        right_site_id
    ]
    .reshape(3, 3)
    .copy()
)


# ------------------------------------------------------------
# Grasp targets
# ------------------------------------------------------------

left_grasp_target = np.array([
    0.0,
    0.135,
    0.56,
])

right_grasp_target = np.array([
    0.0,
    -0.135,
    0.56,
])


# ------------------------------------------------------------
# Solve grasp IK
# ------------------------------------------------------------

left_ik_result = solve_pose_ik(
    model=model,
    data=data,
    joint_names=left_joint_names,
    site_name="left_ee_site",
    target_position=left_grasp_target,
    target_rotation=left_rotation,
    initial_q=initial_q,
)


right_ik_result = solve_pose_ik(
    model=model,
    data=data,
    joint_names=right_joint_names,
    site_name="right_ee_site",
    target_position=right_grasp_target,
    target_rotation=right_rotation,
    initial_q=initial_q,
)


# ------------------------------------------------------------
# Extract the 7-DOF joint solution
#
# solve_pose_ik may return additional diagnostic information,
# so normalize the result here instead of changing the reusable
# IK controller.
# ------------------------------------------------------------

def extract_joint_solution(
    result,
    label,
):

    if isinstance(
        result,
        tuple,
    ):
        q = result[0]

    else:
        q = result


    q = np.asarray(
        q,
        dtype=float,
    ).reshape(-1)


    if q.size != 7:

        raise ValueError(
            f"{label} IK solution has "
            f"{q.size} elements instead of 7. "
            f"Raw result: {result}"
        )


    return q


left_grasp_q = extract_joint_solution(
    left_ik_result,
    "Left",
)

right_grasp_q = extract_joint_solution(
    right_ik_result,
    "Right",
)


print(
    "\nLeft grasp q:",
    left_grasp_q,
)

print(
    "Right grasp q:",
    right_grasp_q,
)

# ------------------------------------------------------------
# Put BOTH arms simultaneously at solved grasp pose
# ------------------------------------------------------------

set_arm_qpos(
    left_joint_names,
    left_grasp_q,
)

set_arm_qpos(
    right_joint_names,
    right_grasp_q,
)


# Keep fingers open

for prefix in [
    "left",
    "right",
]:

    for finger_index in [
        1,
        2,
    ]:

        joint_name = (
            f"{prefix}_finger_joint"
            f"{finger_index}"
        )

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_id = model.jnt_qposadr[
            joint_id
        ]

        data.qpos[
            qpos_id
        ] = 0.04


mujoco.mj_forward(
    model,
    data,
)


# ------------------------------------------------------------
# Print reference sites
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("REFERENCE POSITIONS AT EXACT IK GRASP POSE")
print("=" * 80)


for name in [
    "left_ee_site",
    "right_ee_site",
    "left_grasp_site",
    "right_grasp_site",
]:

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        name,
    )

    print(
        f"{name}:",
        data.site_xpos[
            site_id
        ].copy(),
    )


# ------------------------------------------------------------
# Inspect collision geoms belonging to finger bodies
# ------------------------------------------------------------

finger_body_names = [
    "left_left_finger",
    "left_right_finger",
    "right_left_finger",
    "right_right_finger",
]


for body_name in finger_body_names:

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )


    print("\n")
    print("=" * 80)
    print(body_name)
    print("=" * 80)


    for geom_id in range(
        model.ngeom
    ):

        if (
            model.geom_bodyid[
                geom_id
            ]
            != body_id
        ):
            continue


        # We care about physical collision geometry.
        if (
            model.geom_contype[
                geom_id
            ] == 0
            and
            model.geom_conaffinity[
                geom_id
            ] == 0
        ):
            continue


        geom_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )

        position = (
            data.geom_xpos[
                geom_id
            ].copy()
        )


        print(
            "\nGeom ID:",
            geom_id,
        )

        print(
            "Name:",
            geom_name,
        )

        print(
            "Type:",
            model.geom_type[
                geom_id
            ],
        )

        print(
            "World position:",
            position,
        )

        print(
            "Size:",
            model.geom_size[
                geom_id
            ].copy(),
        )

        print(
            "contype:",
            model.geom_contype[
                geom_id
            ],
        )

        print(
            "conaffinity:",
            model.geom_conaffinity[
                geom_id
            ],
        )

