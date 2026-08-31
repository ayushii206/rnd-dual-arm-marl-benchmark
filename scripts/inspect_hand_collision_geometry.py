from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.controllers.ik_controller import (
    solve_pose_ik,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "assets/models/dual_panda/transport_scene.xml"
)


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    model
)


# ------------------------------------------------------------
# Initial configuration
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


left_joint_names = [
    f"left_joint{i}"
    for i in range(1, 8)
]

right_joint_names = [
    f"right_joint{i}"
    for i in range(1, 8)
]


def get_qpos_ids(
    joint_names,
):
    ids = []

    for name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        ids.append(
            model.jnt_qposadr[
                joint_id
            ]
        )

    return ids


def set_qpos(
    qpos_ids,
    q,
):
    for qpos_id, value in zip(
        qpos_ids,
        q,
    ):
        data.qpos[
            qpos_id
        ] = value


def print_body_geoms(
    body_name,
    ee_site_name,
):
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    ee_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        ee_site_name,
    )

    ee_position = data.site_xpos[
        ee_site_id
    ].copy()

    ee_rotation = (
        data.site_xmat[
            ee_site_id
        ]
        .reshape(
            3,
            3,
        )
        .copy()
    )

    first_geom = model.body_geomadr[
        body_id
    ]

    number_of_geoms = model.body_geomnum[
        body_id
    ]

    print("\n")
    print("=" * 70)
    print(
        f"BODY: {body_name}"
    )
    print("=" * 70)

    print(
        "Body ID:",
        body_id,
    )

    print(
        "EE site:",
        ee_site_name,
    )

    print(
        "EE position:",
        ee_position,
    )

    print(
        "Number of attached geoms:",
        number_of_geoms,
    )

    for offset in range(
        number_of_geoms
    ):

        geom_id = (
            first_geom
            + offset
        )

        geom_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )

        geom_type = model.geom_type[
            geom_id
        ]

        geom_world_position = (
            data.geom_xpos[
                geom_id
            ].copy()
        )

        geom_local_position = (
            model.geom_pos[
                geom_id
            ].copy()
        )

        geom_size = (
            model.geom_size[
                geom_id
            ].copy()
        )

        # Position of geom centre relative to EE,
        # expressed in world coordinates.
        relative_world = (
            geom_world_position
            - ee_position
        )

        # Same vector expressed in the EE's
        # local coordinate frame.
        relative_ee = (
            ee_rotation.T
            @ relative_world
        )

        print("\n----------------------------------------")

        print(
            "Geom ID:",
            geom_id,
        )

        print(
            "Geom name:",
            geom_name,
        )

        print(
            "Geom type:",
            geom_type,
        )

        print(
            "Local geom position:",
            geom_local_position,
        )

        print(
            "World geom position:",
            geom_world_position,
        )

        print(
            "Geom size:",
            geom_size,
        )

        print(
            "Geom centre - EE (world):",
            relative_world,
        )

        print(
            "Geom centre - EE (EE local):",
            relative_ee,
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


# ------------------------------------------------------------
# Put arms in nominal configuration to get nominal rotations
# ------------------------------------------------------------

left_qpos_ids = get_qpos_ids(
    left_joint_names
)

right_qpos_ids = get_qpos_ids(
    right_joint_names
)


set_qpos(
    left_qpos_ids,
    initial_q,
)

set_qpos(
    right_qpos_ids,
    initial_q,
)

mujoco.mj_forward(
    model,
    data,
)


left_ee_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "left_ee_site",
)

right_ee_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_ee_site",
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


# ------------------------------------------------------------
# Exact Cartesian grasp targets
# ------------------------------------------------------------

left_grasp_position = np.array([
    0.0,
    0.135,
    0.56,
])

right_grasp_position = np.array([
    0.0,
    -0.135,
    0.56,
])


left_grasp_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=left_joint_names,
    site_name="left_ee_site",
    target_position=left_grasp_position,
    target_rotation=left_target_rotation,
    initial_q=initial_q,
)


right_grasp_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=right_joint_names,
    site_name="right_ee_site",
    target_position=right_grasp_position,
    target_rotation=right_target_rotation,
    initial_q=initial_q,
)


# ------------------------------------------------------------
# Set the exact IK solution
# ------------------------------------------------------------

set_qpos(
    left_qpos_ids,
    left_grasp_q,
)

set_qpos(
    right_qpos_ids,
    right_grasp_q,
)


# Fully open both grippers.
left_finger1_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "left_finger_joint1",
)

left_finger2_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "left_finger_joint2",
)

right_finger1_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "right_finger_joint1",
)

right_finger2_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "right_finger_joint2",
)


for joint_id in [
    left_finger1_id,
    left_finger2_id,
    right_finger1_id,
    right_finger2_id,
]:

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
# Print exact pose
# ------------------------------------------------------------

print(
    "\nLeft grasp q:",
    left_grasp_q,
)

print(
    "Right grasp q:",
    right_grasp_q,
)

print(
    "\nLeft EE:",
    data.site_xpos[
        left_ee_site_id
    ],
)

print(
    "Right EE:",
    data.site_xpos[
        right_ee_site_id
    ],
)


# ------------------------------------------------------------
# Inspect palm/hand collision geometry
# ------------------------------------------------------------

print_body_geoms(
    "left_hand",
    "left_ee_site",
)

print_body_geoms(
    "right_hand",
    "right_ee_site",
)
