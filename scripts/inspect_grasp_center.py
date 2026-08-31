from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = Path(
    "assets/models/dual_panda/transport_scene.xml"
)


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)


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
# Set nominal arm configuration
# ------------------------------------------------------------

for prefix in [
    "left",
    "right",
]:

    for i, value in enumerate(
        initial_q,
        start=1,
    ):

        joint_name = (
            f"{prefix}_joint{i}"
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
        ] = value


# ------------------------------------------------------------
# Open both grippers directly in qpos
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
# Helper
# ------------------------------------------------------------

def print_gripper_geometry(prefix):

    site_name = (
        f"{prefix}_ee_site"
    )

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    ee_position = (
        data.site_xpos[
            site_id
        ].copy()
    )

    ee_rotation = (
        data.site_xmat[
            site_id
        ].reshape(
            3,
            3,
        ).copy()
    )


    print("\n")
    print("=" * 70)
    print(
        prefix.upper(),
        "GRIPPER"
    )
    print("=" * 70)

    print(
        "EE position:",
        ee_position,
    )


    finger_body_names = [
        f"{prefix}_left_finger",
        f"{prefix}_right_finger",
    ]


    body_positions = []


    for body_name in finger_body_names:

        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        body_position = (
            data.xpos[
                body_id
            ].copy()
        )

        body_positions.append(
            body_position
        )

        relative_world = (
            body_position
            - ee_position
        )

        relative_local = (
            ee_rotation.T
            @ relative_world
        )

        print(
            "\nBody:",
            body_name,
        )

        print(
            "World position:",
            body_position,
        )

        print(
            "Relative to EE (world):",
            relative_world,
        )

        print(
            "Relative to EE (local):",
            relative_local,
        )


    midpoint = (
        body_positions[0]
        + body_positions[1]
    ) / 2.0


    midpoint_world_offset = (
        midpoint
        - ee_position
    )


    midpoint_local_offset = (
        ee_rotation.T
        @ midpoint_world_offset
    )


    print("\n")
    print(
        "Finger body midpoint:",
        midpoint,
    )

    print(
        "Midpoint - EE (world):",
        midpoint_world_offset,
    )

    print(
        "Midpoint - EE (local):",
        midpoint_local_offset,
    )

    print(
        "Distance EE -> midpoint:",
        np.linalg.norm(
            midpoint_world_offset
        ),
    )


print_gripper_geometry(
    "left"
)

print_gripper_geometry(
    "right"
)
