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


# ------------------------------------------------------------
# Nominal robot configuration
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
# Put both arms into nominal configuration
# ------------------------------------------------------------

for prefix in ["left", "right"]:

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

        data.qpos[qpos_id] = value


mujoco.mj_forward(
    model,
    data,
)


# ------------------------------------------------------------
# Inspect each gripper
# ------------------------------------------------------------

for prefix in ["left", "right"]:

    print("\n")
    print("=" * 60)
    print(f"{prefix.upper()} GRIPPER")
    print("=" * 60)

    # --------------------------------------------------------
    # End-effector site
    # --------------------------------------------------------

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
        ]
        .reshape(3, 3)
        .copy()
    )

    print("\nEnd-effector position:")
    print(ee_position)

    print("\nEnd-effector rotation:")
    print(ee_rotation)

    print("\nEE local X axis in world:")
    print(
        ee_rotation[:, 0]
    )

    print("\nEE local Y axis in world:")
    print(
        ee_rotation[:, 1]
    )

    print("\nEE local Z axis in world:")
    print(
        ee_rotation[:, 2]
    )

    # --------------------------------------------------------
    # Finger joints
    # --------------------------------------------------------

    for finger_number in [1, 2]:

        joint_name = (
            f"{prefix}_finger_joint"
            f"{finger_number}"
        )

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id == -1:

            print(
                f"\nJoint not found: "
                f"{joint_name}"
            )

            continue

        qpos_id = model.jnt_qposadr[
            joint_id
        ]

        joint_type = model.jnt_type[
            joint_id
        ]

        local_axis = (
            model.jnt_axis[
                joint_id
            ].copy()
        )

        body_id = model.jnt_bodyid[
            joint_id
        ]

        body_rotation = (
            data.xmat[
                body_id
            ]
            .reshape(3, 3)
            .copy()
        )

        world_axis = (
            body_rotation
            @ local_axis
        )

        print("\n")
        print(
            f"{joint_name}:"
        )

        print(
            "  joint id:",
            joint_id,
        )

        print(
            "  qpos id:",
            qpos_id,
        )

        print(
            "  joint type:",
            joint_type,
        )

        print(
            "  local axis:",
            local_axis,
        )

        print(
            "  world axis:",
            world_axis,
        )

        print(
            "  joint range:",
            model.jnt_range[
                joint_id
            ],
        )

        print(
            "  current qpos:",
            data.qpos[
                qpos_id
            ],
        )


# ------------------------------------------------------------
# Inspect actuators
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("GRIPPER ACTUATORS")
print("=" * 60)


for actuator_id in range(
    model.nu
):

    actuator_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_id,
    )

    if actuator_name is None:
        continue

    if (
        "finger" in actuator_name.lower()
        or "gripper" in actuator_name.lower()
        or actuator_id in [7, 15]
    ):

        print(
            f"\nID {actuator_id}: "
            f"{actuator_name}"
        )

        print(
            "  ctrl range:",
            model.actuator_ctrlrange[
                actuator_id
            ],
        )

