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


for prefix in [
    "left",
    "right",
]:

    joint_names = [
        f"{prefix}_joint{i}"
        for i in range(1, 8)
    ]

    for name, value in zip(
        joint_names,
        initial_q,
    ):

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        qpos_id = (
            model.jnt_qposadr[
                joint_id
            ]
        )

        data.qpos[
            qpos_id
        ] = value


mujoco.mj_forward(
    model,
    data,
)


for site_name in [
    "left_ee_site",
    "right_ee_site",
]:

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    position = (
        data.site_xpos[
            site_id
        ].copy()
    )

    rotation = (
        data.site_xmat[
            site_id
        ]
        .reshape(3, 3)
        .copy()
    )

    print(
        f"\n{site_name}"
    )

    print(
        "Position:"
    )

    print(
        position
    )

    print(
        "Rotation matrix:"
    )

    print(
        rotation
    )
