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
# Nominal arm configuration
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
# Print relevant body names first
# ------------------------------------------------------------

print("\nRelevant bodies:")

for body_id in range(
    model.nbody
):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
    )

    if name is None:
        continue

    lower_name = name.lower()

    if (
        "finger" in lower_name
        or "hand" in lower_name
    ):

        print(
            body_id,
            name,
        )


# ------------------------------------------------------------
# Inspect each gripper
# ------------------------------------------------------------

for prefix in ["left", "right"]:

    print("\n")
    print("=" * 60)
    print(
        f"{prefix.upper()} GRIPPER"
    )
    print("=" * 60)

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

    print(
        "\nEE position:"
    )
    print(
        ee_position
    )

    print(
        "\nEE rotation:"
    )
    print(
        ee_rotation
    )

    # --------------------------------------------------------
    # Find finger bodies automatically
    # --------------------------------------------------------

    finger_body_ids = []

    for body_id in range(
        model.nbody
    ):

        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        if body_name is None:
            continue

        lower_name = (
            body_name.lower()
        )

        if (
            lower_name.startswith(
                f"{prefix}_"
            )
            and "finger" in lower_name
        ):

            finger_body_ids.append(
                body_id
            )

    for body_id in finger_body_ids:

        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        body_position = (
            data.xpos[
                body_id
            ].copy()
        )

        world_offset = (
            body_position
            - ee_position
        )

        # Convert world-space offset into
        # the EE site's local coordinates.
        local_offset = (
            ee_rotation.T
            @ world_offset
        )

        print("\n")
        print(
            f"Body: {body_name}"
        )

        print(
            "World position:"
        )
        print(
            body_position
        )

        print(
            "Offset from EE "
            "(world frame):"
        )
        print(
            world_offset
        )

        print(
            "Offset from EE "
            "(EE local frame):"
        )
        print(
            local_offset
        )


# ------------------------------------------------------------
# Inspect all geoms attached to finger bodies
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("FINGER-BODY GEOMS")
print("=" * 60)


for prefix in ["left", "right"]:

    print("\n")
    print(
        f"{prefix.upper()} FINGER GEOMS"
    )

    print("-" * 60)

    finger_body_ids = []

    for body_id in range(
        model.nbody
    ):

        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        if body_name is None:
            continue

        lower_name = (
            body_name.lower()
        )

        if (
            lower_name.startswith(
                f"{prefix}_"
            )
            and "finger" in lower_name
        ):

            finger_body_ids.append(
                body_id
            )

    for body_id in finger_body_ids:

        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        print("\n")
        print(
            f"Body: {body_name}"
        )

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

            geom_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )

            geom_type = (
                model.geom_type[
                    geom_id
                ]
            )

            geom_position = (
                data.geom_xpos[
                    geom_id
                ].copy()
            )

            geom_rotation = (
                data.geom_xmat[
                    geom_id
                ]
                .reshape(3, 3)
                .copy()
            )

            print(
                "\n  Geom ID:",
                geom_id,
            )

            print(
                "  Name:",
                geom_name,
            )

            print(
                "  Type:",
                geom_type,
            )

            print(
                "  Local position:",
                model.geom_pos[
                    geom_id
                ],
            )

            print(
                "  Size:",
                model.geom_size[
                    geom_id
                ],
            )

            print(
                "  World position:",
                geom_position,
            )

            print(
                "  contype:",
                model.geom_contype[
                    geom_id
                ],
            )

            print(
                "  conaffinity:",
                model.geom_conaffinity[
                    geom_id
                ],
            )

            # ------------------------------------------------
            # Position relative to EE site
            # ------------------------------------------------

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

            world_offset = (
                geom_position
                - ee_position
            )

            local_offset = (
                ee_rotation.T
                @ world_offset
            )

            print(
                "  Offset from EE "
                "(world):",
                world_offset,
            )

            print(
                "  Offset from EE "
                "(EE local):",
                local_offset,
            )
