from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
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

from src.controllers.gravity_compensation import (
    get_joint_dof_ids,
    apply_bias_compensation,
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


# ============================================================
# Configuration
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


LEFT_PREGRASP_POSITION = np.array([
    0.0,
    0.185,
    0.56,
])

RIGHT_PREGRASP_POSITION = np.array([
    0.0,
    -0.185,
    0.56,
])


LEFT_GRASP_POSITION = np.array([
    0.0,
    0.145,
    0.56,
])

RIGHT_GRASP_POSITION = np.array([
    0.0,
    -0.145,
    0.56,
])


# First lift:
# raise both end effectors exactly 2 cm.

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


OPEN_GRIPPER_CTRL = 255.0
CLOSE_GRIPPER_CTRL = 0.0


# ============================================================
# Names
# ============================================================

left_joint_names = [
    f"left_joint{i}"
    for i in range(1, 8)
]

right_joint_names = [
    f"right_joint{i}"
    for i in range(1, 8)
]


# ============================================================
# Helper functions
# ============================================================

def get_qpos_ids(
    joint_names,
):
    qpos_ids = []

    for joint_name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_id = model.jnt_qposadr[
            joint_id
        ]

        qpos_ids.append(
            qpos_id
        )

    return qpos_ids


def get_actuator_ids(
    prefix,
):
    actuator_ids = []

    for actuator_id in range(
        model.nu
    ):

        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        if (
            actuator_name is not None
            and actuator_name.startswith(
                prefix
            )
        ):

            actuator_ids.append(
                actuator_id
            )

    return actuator_ids


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


def command_arm(
    actuator_ids,
    q,
):
    for actuator_id, target in zip(
        actuator_ids,
        q,
    ):

        data.ctrl[
            actuator_id
        ] = target


def get_arm_qpos(
    qpos_ids,
):
    return np.array([
        data.qpos[
            qpos_id
        ]
        for qpos_id in qpos_ids
    ])


def apply_arm_bias_compensation():
    """
    Apply MuJoCo generalized bias-force
    feed-forward compensation to both arms.
    """

    data.qfrc_applied[:] = 0.0

    apply_bias_compensation(
        model,
        data,
        left_dof_ids,
    )

    apply_bias_compensation(
        model,
        data,
        right_dof_ids,
    )


def step_with_arm_targets(
    left_q,
    right_q,
):
    """
    Execute one physics step while maintaining
    both arm joint targets.
    """

    command_arm(
        left_actuator_ids,
        left_q,
    )

    command_arm(
        right_actuator_ids,
        right_q,
    )

    apply_arm_bias_compensation()

    mujoco.mj_step(
        model,
        data,
    )


def interpolate_arm_targets(
    viewer,
    left_start_q,
    left_target_q,
    right_start_q,
    right_target_q,
    steps,
):
    """
    Smoothly interpolate both arm targets.
    """

    for step in range(
        steps
    ):

        alpha = (
            step + 1
        ) / steps

        left_q = (
            (1.0 - alpha)
            * left_start_q
            + alpha
            * left_target_q
        )

        right_q = (
            (1.0 - alpha)
            * right_start_q
            + alpha
            * right_target_q
        )

        step_with_arm_targets(
            left_q,
            right_q,
        )

        viewer.sync()


def print_finger_joint_positions():
    finger_joint_names = [
        "left_finger_joint1",
        "left_finger_joint2",
        "right_finger_joint1",
        "right_finger_joint2",
    ]

    print("\n")
    print("=" * 70)
    print("FINGER JOINT POSITIONS")
    print("=" * 70)

    for joint_name in finger_joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_id = model.jnt_qposadr[
            joint_id
        ]

        print(
            f"{joint_name}: "
            f"{data.qpos[qpos_id]:.6f} m"
        )

    print(
        "=" * 70
    )


def print_object_contacts(
    title,
):
    """
    Print every active contact involving the transport object.

    For each contact:
    - print the MuJoCo contact-frame force
    - transform the force into world coordinates
    - determine the force acting ON the transport object
    - accumulate vertical support force

    This allows us to distinguish normal gripping force
    from the tangential friction force that can actually
    support the object against gravity.
    """

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    found_contact = False

    total_object_force_world = np.zeros(3)
    total_finger_force_world = np.zeros(3)
    total_table_force_world = np.zeros(3)

    total_finger_vertical_force = 0.0
    total_table_vertical_force = 0.0

    finger_contact_count = 0
    table_contact_count = 0

    for contact_id in range(
        data.ncon
    ):

        contact = data.contact[
            contact_id
        ]

        geom1_id = contact.geom1
        geom2_id = contact.geom2

        body1_id = model.geom_bodyid[
            geom1_id
        ]

        body2_id = model.geom_bodyid[
            geom2_id
        ]

        body1_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body1_id,
        )

        body2_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body2_id,
        )

        if (
            body1_name
            != "transport_object"
            and body2_name
            != "transport_object"
        ):
            continue

        found_contact = True

        geom1_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom1_id,
        )

        geom2_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom2_id,
        )

        # ----------------------------------------------------
        # Contact force in MuJoCo contact coordinates
        # ----------------------------------------------------

        contact_wrench = np.zeros(
            6
        )

        mujoco.mj_contactForce(
            model,
            data,
            contact_id,
            contact_wrench,
        )

        contact_force_local = (
            contact_wrench[
                0:3
            ].copy()
        )

        # ----------------------------------------------------
        # Contact frame
        #
        # MuJoCo stores the contact frame as:
        #
        # row 0 = contact normal
        # row 1 = first tangent direction
        # row 2 = second tangent direction
        #
        # Therefore a force expressed in contact coordinates
        # is transformed to world coordinates using frame.T.
        # ----------------------------------------------------

        contact_frame = (
            contact.frame
            .reshape(
                3,
                3,
            )
        )

        # mj_contactForce follows MuJoCo's contact convention:
        # the contact force points from geom1 toward geom2.
        #
        # Therefore the transformed force is the force acting
        # on geom2. The equal-and-opposite force acts on geom1.

        force_world_on_geom2 = (
            contact_frame.T
            @ contact_force_local
        )

        force_world_on_geom1 = (
            -force_world_on_geom2
        )

        # ----------------------------------------------------
        # Determine force acting specifically on the object
        # ----------------------------------------------------

        if (
            body1_name
            == "transport_object"
        ):

            object_force_world = (
                force_world_on_geom1
            )

            other_body_name = (
                body2_name
            )

        else:

            object_force_world = (
                force_world_on_geom2
            )

            other_body_name = (
                body1_name
            )

        total_object_force_world += (
            object_force_world
        )

        # ----------------------------------------------------
        # Classify support source
        # ----------------------------------------------------

        is_finger_contact = (
            other_body_name
            is not None
            and "finger"
            in other_body_name.lower()
        )

        is_table_contact = (
            other_body_name
            == "world"
            and (
                geom1_name
                == "work_table"
                or geom2_name
                == "work_table"
            )
        )

        if is_finger_contact:

            finger_contact_count += 1

            total_finger_force_world += (
                object_force_world
            )

            total_finger_vertical_force += (
                object_force_world[2]
            )

        if is_table_contact:

            table_contact_count += 1

            total_table_force_world += (
                object_force_world
            )

            total_table_vertical_force += (
                object_force_world[2]
            )

        # ----------------------------------------------------
        # Individual contact diagnostic
        # ----------------------------------------------------

        print(
            f"\nContact {contact_id}"
        )

        print(
            "  geom1:",
            geom1_id,
            "/",
            geom1_name,
        )

        print(
            "  body1:",
            body1_name,
        )

        print(
            "  geom2:",
            geom2_id,
            "/",
            geom2_name,
        )

        print(
            "  body2:",
            body2_name,
        )

        print(
            "  distance:",
            contact.dist,
        )

        print(
            "  contact position:",
            contact.pos,
        )

        print(
            "  contact normal:",
            contact_frame[0],
        )
        
        print(
            "  effective contact friction:",
            contact.friction,
        )
        
        print(
            "  contact dimension:",
            contact.dim,
        )

        print(
            "  force in contact frame:",
            contact_force_local,
        )

        print(
            "  normal component:",
            contact_force_local[0],
        )

        print(
            "  tangent component 1:",
            contact_force_local[1],
        )

        print(
            "  tangent component 2:",
            contact_force_local[2],
        )

        print(
            "  force on object [world XYZ]:",
            object_force_world,
        )

        print(
            "  vertical force on object [world Z]:",
            object_force_world[2],
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    if not found_contact:

        print(
            "\nNo contacts involving transport_object."
        )

    else:

        print("\n")
        print("-" * 70)
        print("CONTACT FORCE SUMMARY")
        print("-" * 70)

        print(
            "Finger contact count:",
            finger_contact_count,
        )

        print(
            "Table contact count:",
            table_contact_count,
        )

        print(
            "\nTotal finger force on object [world XYZ]:",
            total_finger_force_world,
        )

        print(
            "Total finger vertical force on object [N]:",
            total_finger_vertical_force,
        )

        print(
            "\nTotal table force on object [world XYZ]:",
            total_table_force_world,
        )

        print(
            "Total table vertical force on object [N]:",
            total_table_vertical_force,
        )

        print(
            "\nTotal contact force on object [world XYZ]:",
            total_object_force_world,
        )

        print(
            "Total contact vertical force on object [N]:",
            total_object_force_world[2],
        )

        print(
            "\nExpected object weight for 2.0 kg [N]:",
            2.0 * 9.81,
        )

    print(
        "\n" + "=" * 70
    )
    
def get_object_support_forces():
    """
    Return the instantaneous vertical contact forces acting
    on the transport object.

    Returns
    -------
    finger_fz : float
        Total world-Z force from finger contacts.

    table_fz : float
        Total world-Z force from table contacts.

    finger_normal_sum : float
        Sum of finger-handle contact normal-force magnitudes.

    finger_contact_count : int
        Number of active finger-object contacts.
    """

    finger_fz = 0.0
    table_fz = 0.0

    finger_normal_sum = 0.0
    finger_contact_count = 0

    for contact_id in range(data.ncon):

        contact = data.contact[
            contact_id
        ]

        geom1_id = contact.geom1
        geom2_id = contact.geom2

        body1_id = model.geom_bodyid[
            geom1_id
        ]

        body2_id = model.geom_bodyid[
            geom2_id
        ]

        body1_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body1_id,
        )

        body2_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body2_id,
        )

        if (
            body1_name != "transport_object"
            and body2_name != "transport_object"
        ):
            continue

        geom1_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom1_id,
        )

        geom2_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom2_id,
        )

        contact_wrench = np.zeros(6)

        mujoco.mj_contactForce(
            model,
            data,
            contact_id,
            contact_wrench,
        )

        contact_force_local = (
            contact_wrench[:3]
        )

        contact_frame = (
            contact.frame.reshape(
                3,
                3,
            )
        )

        # Force acting on geom2.
        force_world_on_geom2 = (
            contact_frame.T
            @ contact_force_local
        )

        # Equal and opposite force on geom1.
        force_world_on_geom1 = (
            -force_world_on_geom2
        )

        if body1_name == "transport_object":

            object_force_world = (
                force_world_on_geom1
            )

            other_body_name = (
                body2_name
            )

        else:

            object_force_world = (
                force_world_on_geom2
            )

            other_body_name = (
                body1_name
            )

        is_finger_contact = (
            other_body_name is not None
            and "finger"
            in other_body_name.lower()
        )

        is_table_contact = (
            other_body_name == "world"
            and (
                geom1_name == "work_table"
                or geom2_name == "work_table"
            )
        )

        if is_finger_contact:

            finger_contact_count += 1

            finger_fz += (
                object_force_world[2]
            )

            finger_normal_sum += abs(
                contact_force_local[0]
            )

        if is_table_contact:

            table_fz += (
                object_force_world[2]
            )

    return (
        finger_fz,
        table_fz,
        finger_normal_sum,
        finger_contact_count,
    )
    
def object_state():
    """
    Return object position, rotation matrix,
    linear velocity and angular velocity.
    """

    object_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "transport_object",
    )

    position = data.xpos[
        object_body_id
    ].copy()

    rotation = (
        data.xmat[
            object_body_id
        ]
        .reshape(
            3,
            3,
        )
        .copy()
    )

    object_joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "object_freejoint",
    )

    dof_address = model.jnt_dofadr[
        object_joint_id
    ]

    linear_velocity = data.qvel[
        dof_address:
        dof_address + 3
    ].copy()

    angular_velocity = data.qvel[
        dof_address + 3:
        dof_address + 6
    ].copy()

    return (
        position,
        rotation,
        linear_velocity,
        angular_velocity,
    )


def rotation_matrix_to_rpy(
    rotation,
):
    """
    Convert a rotation matrix to roll/pitch/yaw.
    Used only for diagnostics.
    """

    pitch = np.arcsin(
        np.clip(
            -rotation[2, 0],
            -1.0,
            1.0,
        )
    )

    roll = np.arctan2(
        rotation[2, 1],
        rotation[2, 2],
    )

    yaw = np.arctan2(
        rotation[1, 0],
        rotation[0, 0],
    )

    return np.array([
        roll,
        pitch,
        yaw,
    ])


def print_object_state(
    title,
):
    (
        position,
        rotation,
        linear_velocity,
        angular_velocity,
    ) = object_state()

    rpy = rotation_matrix_to_rpy(
        rotation
    )

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        "Object position:",
        position,
    )

    print(
        "Object RPY [rad]:",
        rpy,
    )

    print(
        "Object RPY [deg]:",
        np.degrees(
            rpy
        ),
    )

    print(
        "Linear velocity:",
        linear_velocity,
    )

    print(
        "Angular velocity:",
        angular_velocity,
    )

    print(
        "=" * 70
    )

def print_gripper_actuator_state(
    title,
):
    """
    Print the actual MuJoCo actuator state for
    both grippers at the current simulation state.
    """

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    for actuator_id in [
        left_gripper_actuator_id,
        right_gripper_actuator_id,
    ]:

        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        print(
            f"\n{actuator_name}"
        )

        print(
            "ctrl:",
            data.ctrl[
                actuator_id
            ],
        )

        print(
            "actuator length:",
            data.actuator_length[
                actuator_id
            ],
        )

        print(
            "actuator velocity:",
            data.actuator_velocity[
                actuator_id
            ],
        )

        print(
            "actuator force:",
            data.actuator_force[
                actuator_id
            ],
        )

    print(
        "\n" + "=" * 70
    )
    
def print_gripper_transmission_state(
    title,
):
    """
    Inspect how each gripper actuator force is transmitted
    through its fixed tendon to the two finger joint DOFs.
    """

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    grippers = [
        {
            "label": "LEFT GRIPPER",
            "actuator_id": left_gripper_actuator_id,
            "tendon_name": "left_split",
            "finger_joint_names": [
                "left_finger_joint1",
                "left_finger_joint2",
            ],
        },
        {
            "label": "RIGHT GRIPPER",
            "actuator_id": right_gripper_actuator_id,
            "tendon_name": "right_split",
            "finger_joint_names": [
                "right_finger_joint1",
                "right_finger_joint2",
            ],
        },
    ]

    tendon_jacobian_values = np.asarray(
        data.ten_J
    )

    print(
        "data.ten_J shape:",
        tendon_jacobian_values.shape,
    )

    print(
        "model.ten_J_rownnz:",
        np.asarray(
            model.ten_J_rownnz
        ),
    )

    print(
        "model.ten_J_rowadr:",
        np.asarray(
            model.ten_J_rowadr
        ),
    )

    print(
        "model.ten_J_colind:",
        np.asarray(
            model.ten_J_colind
        ),
    )

    for gripper in grippers:

        print("\n")
        print("-" * 70)
        print(
            gripper["label"]
        )
        print("-" * 70)

        actuator_id = gripper[
            "actuator_id"
        ]

        tendon_name = gripper[
            "tendon_name"
        ]

        tendon_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            tendon_name,
        )

        print(
            "Actuator ID:",
            actuator_id,
        )

        print(
            "Actuator force:",
            data.actuator_force[
                actuator_id
            ],
        )

        print(
            "Actuator length:",
            data.actuator_length[
                actuator_id
            ],
        )

        print(
            "Actuator velocity:",
            data.actuator_velocity[
                actuator_id
            ],
        )

        print(
            "\nTendon name:",
            tendon_name,
        )

        print(
            "Tendon ID:",
            tendon_id,
        )

        print(
            "Tendon length:",
            data.ten_length[
                tendon_id
            ],
        )

        print(
            "Tendon velocity:",
            data.ten_velocity[
                tendon_id
            ],
        )

# ----------------------------------------------------
# Reconstruct this tendon Jacobian row from MuJoCo's
# sparse tendon Jacobian representation.
#
# In MuJoCo 3.8.1:
#
# data.ten_J
#     stores only the non-zero Jacobian values.
#
# model.ten_J_rowadr
#     gives the starting index for each tendon row.
#
# model.ten_J_rownnz
#     gives the number of non-zero entries in the row.
#
# model.ten_J_colind
#     gives the corresponding DOF column indices.
# ----------------------------------------------------

        row_start = int(
            model.ten_J_rowadr[
                tendon_id
            ]
        )

        row_nnz = int(
            model.ten_J_rownnz[
                tendon_id
            ]
        )

        row_end = (
            row_start
            + row_nnz
        )

        row_columns = np.asarray(
            model.ten_J_colind[
                row_start:row_end
            ],
            dtype=int,
        )

        row_values = np.asarray(
            data.ten_J[
                row_start:row_end
            ]
        )

        tendon_jacobian_row = np.zeros(
            model.nv
        )

        tendon_jacobian_row[
            row_columns
        ] = row_values

        print(
            "\nSparse tendon Jacobian:"
        )

        print(
            "  row start:",
            row_start,
        )

        print(
            "  non-zero count:",
            row_nnz,
        )

        print(
            "  DOF columns:",
            row_columns,
        )

        print(
            "  values:",
            row_values,
        )
        
        print(
            "\nFinger transmission:"
        )

        actuator_force = (
            data.actuator_force[
                actuator_id
            ]
        )

        for joint_name in gripper[
            "finger_joint_names"
        ]:

            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            qpos_id = model.jnt_qposadr[
                joint_id
            ]

            dof_id = model.jnt_dofadr[
                joint_id
            ]

            tendon_coefficient = (
                tendon_jacobian_row[
                    dof_id
                ]
            )

            predicted_generalized_force = (
                actuator_force
                * tendon_coefficient
            )

            actual_generalized_force = (
                data.qfrc_actuator[
                    dof_id
                ]
            )

            print(
                f"\n{joint_name}"
            )

            print(
                "  Joint ID:",
                joint_id,
            )

            print(
                "  qpos ID:",
                qpos_id,
            )

            print(
                "  DOF ID:",
                dof_id,
            )

            print(
                "  joint position:",
                data.qpos[
                    qpos_id
                ],
            )

            print(
                "  tendon Jacobian coefficient:",
                tendon_coefficient,
            )

            print(
                "  actuator_force * coefficient:",
                predicted_generalized_force,
            )

            print(
                "  data.qfrc_actuator:",
                actual_generalized_force,
            )

        print(
            "\nFull non-zero tendon Jacobian entries:"
        )

        nonzero_dofs = np.where(
            np.abs(
                tendon_jacobian_row
            )
            > 1e-10
        )[0]

        for dof_id in nonzero_dofs:

            print(
                f"  DOF {dof_id}: "
                f"{tendon_jacobian_row[dof_id]}"
            )

    print(
        "\n" + "=" * 70
    )
    
# ============================================================
# IDs
# ============================================================

left_qpos_ids = get_qpos_ids(
    left_joint_names
)

right_qpos_ids = get_qpos_ids(
    right_joint_names
)


left_actuator_ids = get_actuator_ids(
    "left_actuator"
)[:7]

right_actuator_ids = get_actuator_ids(
    "right_actuator"
)[:7]


left_gripper_actuator_id = (
    mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "left_actuator8",
    )
)

right_gripper_actuator_id = (
    mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "right_actuator8",
    )
)


left_dof_ids = get_joint_dof_ids(
    model,
    left_actuator_ids,
)

right_dof_ids = get_joint_dof_ids(
    model,
    right_actuator_ids,
)


print(
    "Left arm actuator IDs:",
    left_actuator_ids,
)

print(
    "Right arm actuator IDs:",
    right_actuator_ids,
)

print(
    "Left gripper actuator ID:",
    left_gripper_actuator_id,
)

print(
    "Right gripper actuator ID:",
    right_gripper_actuator_id,
)


# ============================================================
# Initial arm pose
# ============================================================

set_qpos(
    left_qpos_ids,
    INITIAL_ARM_Q,
)

set_qpos(
    right_qpos_ids,
    INITIAL_ARM_Q,
)

mujoco.mj_forward(
    model,
    data,
)


# ============================================================
# Preserve nominal EE orientations
# ============================================================

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


# ============================================================
# Solve IK
# ============================================================

left_pregrasp_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=left_joint_names,
    site_name="left_ee_site",
    target_position=LEFT_PREGRASP_POSITION,
    target_rotation=left_target_rotation,
    initial_q=INITIAL_ARM_Q,
)


right_pregrasp_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=right_joint_names,
    site_name="right_ee_site",
    target_position=RIGHT_PREGRASP_POSITION,
    target_rotation=right_target_rotation,
    initial_q=INITIAL_ARM_Q,
)


left_grasp_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=left_joint_names,
    site_name="left_ee_site",
    target_position=LEFT_GRASP_POSITION,
    target_rotation=left_target_rotation,
    initial_q=left_pregrasp_q,
)


right_grasp_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=right_joint_names,
    site_name="right_ee_site",
    target_position=RIGHT_GRASP_POSITION,
    target_rotation=right_target_rotation,
    initial_q=right_pregrasp_q,
)


left_lift_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=left_joint_names,
    site_name="left_ee_site",
    target_position=LEFT_LIFT_POSITION,
    target_rotation=left_target_rotation,
    initial_q=left_grasp_q,
)


right_lift_q, _, _ = solve_pose_ik(
    model=model,
    data=data,
    joint_names=right_joint_names,
    site_name="right_ee_site",
    target_position=RIGHT_LIFT_POSITION,
    target_rotation=right_target_rotation,
    initial_q=right_grasp_q,
)


print(
    "\nIK solved."
)

print(
    "Left grasp q:",
    left_grasp_q,
)

print(
    "Right grasp q:",
    right_grasp_q,
)

print(
    "Left lift q:",
    left_lift_q,
)

print(
    "Right lift q:",
    right_lift_q,
)


# ============================================================
# Reset simulation after IK
# ============================================================

mujoco.mj_resetData(
    model,
    data,
)

set_qpos(
    left_qpos_ids,
    INITIAL_ARM_Q,
)

set_qpos(
    right_qpos_ids,
    INITIAL_ARM_Q,
)

mujoco.mj_forward(
    model,
    data,
)


command_arm(
    left_actuator_ids,
    INITIAL_ARM_Q,
)

command_arm(
    right_actuator_ids,
    INITIAL_ARM_Q,
)

data.ctrl[
    left_gripper_actuator_id
] = OPEN_GRIPPER_CTRL

data.ctrl[
    right_gripper_actuator_id
] = OPEN_GRIPPER_CTRL


# ============================================================
# Viewer / experiment
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    # --------------------------------------------------------
    # Open grippers
    # --------------------------------------------------------

    print(
        "\nOpening grippers..."
    )

    for _ in range(
        500
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = OPEN_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = OPEN_GRIPPER_CTRL

        step_with_arm_targets(
            INITIAL_ARM_Q,
            INITIAL_ARM_Q,
        )

        viewer.sync()


    # --------------------------------------------------------
    # Move to pre-grasp
    # --------------------------------------------------------

    print(
        "Moving to pre-grasp..."
    )

    interpolate_arm_targets(
        viewer,
        INITIAL_ARM_Q,
        left_pregrasp_q,
        INITIAL_ARM_Q,
        right_pregrasp_q,
        2000,
    )

    print(
        "Pre-grasp reached."
    )


    # --------------------------------------------------------
    # Settle at pre-grasp
    # --------------------------------------------------------

    for _ in range(
        500
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = OPEN_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = OPEN_GRIPPER_CTRL

        step_with_arm_targets(
            left_pregrasp_q,
            right_pregrasp_q,
        )

        viewer.sync()


    # --------------------------------------------------------
    # Move into grasp pose
    # --------------------------------------------------------

    print(
        "Moving into grasp..."
    )

    interpolate_arm_targets(
        viewer,
        left_pregrasp_q,
        left_grasp_q,
        right_pregrasp_q,
        right_grasp_q,
        1500,
    )

    print(
        "Grasp pose reached."
    )


    # --------------------------------------------------------
    # Settle before closing
    # --------------------------------------------------------

    print(
        "Settling at grasp pose..."
    )

    for _ in range(
        1000
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = OPEN_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = OPEN_GRIPPER_CTRL

        step_with_arm_targets(
            left_grasp_q,
            right_grasp_q,
        )

        viewer.sync()


    print(
        "Grasp pose settled."
    )


    # --------------------------------------------------------
    # State before closing
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        data,
    )

    print_object_state(
        "OBJECT STATE BEFORE GRIPPER CLOSING"
    )

    print_object_contacts(
        "CONTACTS BEFORE GRIPPER CLOSING"
    )


    # --------------------------------------------------------
    # Close both grippers
    # --------------------------------------------------------

    print(
        "\nClosing grippers..."
    )

    for _ in range(
        2000
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        step_with_arm_targets(
            left_grasp_q,
            right_grasp_q,
        )

        viewer.sync()


    print(
        "Grippers closed."
    )


    # --------------------------------------------------------
    # Settle closed grasp
    # --------------------------------------------------------

    print(
        "Settling closed grasp..."
    )

    for _ in range(
        1000
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        step_with_arm_targets(
            left_grasp_q,
            right_grasp_q,
        )

        viewer.sync()


    mujoco.mj_forward(
        model,
        data,
    )


    print_finger_joint_positions()

    print_gripper_actuator_state(
        "GRIPPER ACTUATOR STATE BEFORE LIFT"
    )
    
    print_gripper_transmission_state(
    "GRIPPER TRANSMISSION STATE BEFORE LIFT"
)

    print_object_state(
        "OBJECT STATE BEFORE LIFT"
    )

    print_object_contacts(
        "CONTACTS BEFORE LIFT"
    )


    object_position_before_lift, _, _, _ = (
        object_state()
    )


    # --------------------------------------------------------
    # Cooperative 2 cm lift
    # --------------------------------------------------------

    print(
        "\nStarting cooperative 2 cm lift..."
    )

# --------------------------------------------------------
# Instrumented cooperative lift
#
# Record how vertical support changes WHILE the fingers
# move upward, not only after the lift has finished.
# --------------------------------------------------------

    LIFT_STEPS = 2500
    LOG_INTERVAL = 100

    object_center_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "object_center",
    )
    
    object_freejoint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "object_freejoint",
    )

    object_freejoint_dof_adr = model.jnt_dofadr[
        object_freejoint_id
    ]

    max_finger_fz = -np.inf
    max_finger_fz_step = -1

    max_finger_normal_sum = -np.inf
    max_finger_normal_step = -1

    min_table_fz = np.inf
    min_table_fz_step = -1

    max_object_z = -np.inf
    max_object_z_step = -1

    initial_object_z = float(
        data.site_xpos[
            object_center_site_id,
            2,
        ]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DYNAMIC LIFT FORCE DIAGNOSTIC"
    )

    print(
        "=" * 70
    )

    print(
        "Initial object Z:",
        initial_object_z,
    )

    print(
        "\n"
        "step | progress | object_z | "
        "delta_z | object_vz | finger_Fz | table_Fz | "
        "normal_sum | finger_contacts"
    )

    print(
        "-" * 110
    )

    for step_index in range(
        LIFT_STEPS
    ):

        # Progress goes from approximately 0 to exactly 1.
        alpha = (
            (step_index + 1)
            / LIFT_STEPS
        )

        left_target_q = (
            left_grasp_q
            + alpha
            * (
                left_lift_q
                - left_grasp_q
            )
        )

        right_target_q = (
            right_grasp_q
            + alpha
            * (
                right_lift_q
                - right_grasp_q
            )
        )

        # Keep both grippers fully closed throughout
        # the entire upward trajectory.
        data.ctrl[
            left_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        # This uses the same arm stepping / bias compensation
        # already used everywhere else in this experiment.
        step_with_arm_targets(
            left_target_q,
            right_target_q,
        )

        viewer.sync()

        # ----------------------------------------------------
        # Measure contact support AFTER this simulation step.
        # ----------------------------------------------------

        (
            finger_fz,
            table_fz,
            finger_normal_sum,
            finger_contact_count,
        ) = get_object_support_forces()

        object_z = float(
            data.site_xpos[
                object_center_site_id,
                2,
            ]
        )

        object_delta_z = (
            object_z
            - initial_object_z
        )
        
        object_vz = float(
            data.qvel[
                object_freejoint_dof_adr + 2
            ]
        )

        # ----------------------------------------------------
        # Track extrema over the complete 2500-step motion.
        # ----------------------------------------------------

        if (
            finger_fz
            > max_finger_fz
        ):

            max_finger_fz = (
                finger_fz
            )

            max_finger_fz_step = (
                step_index + 1
            )

        if (
            finger_normal_sum
            > max_finger_normal_sum
        ):

            max_finger_normal_sum = (
                finger_normal_sum
            )

            max_finger_normal_step = (
                step_index + 1
            )

        if (
            table_fz
            < min_table_fz
        ):

            min_table_fz = (
                table_fz
            )

            min_table_fz_step = (
                step_index + 1
            )

        if (
            object_z
            > max_object_z
        ):

            max_object_z = (
                object_z
            )

            max_object_z_step = (
                step_index + 1
            )

        # ----------------------------------------------------
        # Print every 100 steps, plus first and final step.
        # ----------------------------------------------------

        if (
            step_index == 0
            or (
                step_index + 1
            ) % LOG_INTERVAL == 0
            or step_index
            == LIFT_STEPS - 1
        ):

            print(
                f"{step_index + 1:4d} | "
                f"{alpha:8.4f} | "
                f"{object_z:9.6f} | "
                f"{object_delta_z:+9.6f} | "
                f"{object_vz:+10.6f} | "
                f"{finger_fz:+9.4f} | "
                f"{table_fz:+9.4f} | "
                f"{finger_normal_sum:10.4f} | "
                f"{finger_contact_count:3d}"
            )

    print(
        "\n"
        + "-" * 70
    )

    print(
        "DYNAMIC LIFT SUMMARY"
    )

    print(
        "-" * 70
    )

    print(
        "Initial object Z:",
        initial_object_z,
    )

    print(
        "Final object Z:",
        float(
            data.site_xpos[
                object_center_site_id,
                2,
            ]
        ),
    )
    
    print(
        "Object vertical velocity at end of lift [m/s]:",
        float(
            data.qvel[
                object_freejoint_dof_adr + 2
            ]
        ),
    )

    print(
        "Maximum object Z:",
        max_object_z,
    )

    print(
        "Maximum object vertical displacement:",
        max_object_z
        - initial_object_z,
    )

    print(
        "Maximum object Z occurred at step:",
        max_object_z_step,
    )

    print(
        "\nMaximum finger vertical force [N]:",
        max_finger_fz,
    )

    print(
        "Maximum finger vertical force occurred at step:",
        max_finger_fz_step,
    )

    print(
        "\nMaximum summed finger normal force [N]:",
        max_finger_normal_sum,
    )

    print(
        "Maximum summed finger normal force occurred at step:",
        max_finger_normal_step,
    )

    print(
        "\nMinimum table vertical support [N]:",
        min_table_fz,
    )

    print(
        "Minimum table support occurred at step:",
        min_table_fz_step,
    )

    print(
        "\nObject weight [N]:",
        2.0 * 9.81,
    )

    print(
        "=" * 70
    )

    print(
        "Lift target reached."
    )


    # --------------------------------------------------------
    # Hold at lifted position
    # --------------------------------------------------------

    print(
        "Holding lifted pose..."
    )

    # --------------------------------------------------------
    # Instrumented lifted-pose hold
    #
    # The object is off the table at the end of the lift.
    # Measure exactly when and how it slips back downward.
    # --------------------------------------------------------

    HOLD_STEPS = 7500
    HOLD_LOG_INTERVAL = 50

    hold_initial_object_z = float(
        data.site_xpos[
            object_center_site_id,
            2,
        ]
    )

    min_hold_object_z = (
        hold_initial_object_z
    )

    min_hold_object_z_step = 0

    max_hold_finger_fz = -np.inf
    min_hold_finger_fz = np.inf

    first_table_contact_step = None

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STATIC HOLD FORCE DIAGNOSTIC"
    )

    print(
        "=" * 70
    )

    print(
        "Object Z at start of hold:",
        hold_initial_object_z,
    )

    print(
        "\n"
        "step | object_z | delta_from_hold_start | "
        "object_vz | finger_Fz | table_Fz | "
        "normal_sum | finger_contacts"
    )

    print(
        "-" * 115
    )

    for hold_step in range(
        HOLD_STEPS
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        step_with_arm_targets(
            left_lift_q,
            right_lift_q,
        )

        viewer.sync()

        (
            finger_fz,
            table_fz,
            finger_normal_sum,
            finger_contact_count,
        ) = get_object_support_forces()

        object_z = float(
            data.site_xpos[
                object_center_site_id,
                2,
            ]
        )

        delta_from_hold_start = (
            object_z
            - hold_initial_object_z
        )
        
        object_vz = float(
            data.qvel[
                object_freejoint_dof_adr + 2
            ]
        )

        if object_z < min_hold_object_z:

            min_hold_object_z = (
                object_z
            )

            min_hold_object_z_step = (
                hold_step + 1
            )

        max_hold_finger_fz = max(
            max_hold_finger_fz,
            finger_fz,
        )

        min_hold_finger_fz = min(
            min_hold_finger_fz,
            finger_fz,
        )

        # Detect first meaningful return of table support.
        if (
            first_table_contact_step is None
            and table_fz > 0.01
        ):

            first_table_contact_step = (
                hold_step + 1
            )

        if (
            hold_step == 0
            or (
                hold_step + 1
            ) % HOLD_LOG_INTERVAL == 0
            or hold_step
            == HOLD_STEPS - 1
        ):

            print(
                f"{hold_step + 1:4d} | "
                f"{object_z:9.6f} | "
                f"{delta_from_hold_start:+12.6f} | "
                f"{object_vz:+10.6f} | "
                f"{finger_fz:+9.4f} | "
                f"{table_fz:+9.4f} | "
                f"{finger_normal_sum:10.4f} | "
                f"{finger_contact_count:3d}"
            )

    print(
        "\n"
        + "-" * 70
    )

    print(
        "STATIC HOLD SUMMARY"
    )

    print(
        "-" * 70
    )

    print(
        "Object Z at start of hold:",
        hold_initial_object_z,
    )
    
    print(
        "Object vertical velocity at end of hold [m/s]:",
        float(
            data.qvel[
                object_freejoint_dof_adr + 2
            ]
        ),
    )

    print(
        "Object Z at end of hold:",
        float(
            data.site_xpos[
                object_center_site_id,
                2,
            ]
        ),
    )

    print(
        "Total object drop during hold:",
        float(
            data.site_xpos[
                object_center_site_id,
                2,
            ]
        )
        - hold_initial_object_z,
    )

    print(
        "Minimum object Z during hold:",
        min_hold_object_z,
    )

    print(
        "Minimum object Z occurred at step:",
        min_hold_object_z_step,
    )

    print(
        "First table support step:",
        first_table_contact_step,
    )

    print(
        "Maximum finger Fz during hold [N]:",
        max_hold_finger_fz,
    )

    print(
        "Minimum finger Fz during hold [N]:",
        min_hold_finger_fz,
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Final diagnostics
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        data,
    )


    object_position_after_lift, _, _, _ = (
        object_state()
    )


    print_object_state(
        "OBJECT STATE AFTER LIFT"
    )

    print_object_contacts(
        "CONTACTS AFTER LIFT"
    )

    print_finger_joint_positions()
    
    print_gripper_actuator_state(
        "GRIPPER ACTUATOR STATE AFTER LIFT"
    )
    
    print_gripper_transmission_state(
        "GRIPPER TRANSMISSION STATE AFTER LIFT"
    )

    print("\n")
    print("=" * 70)
    print("COOPERATIVE LIFT RESULT")
    print("=" * 70)

    print(
        "Object position before lift:",
        object_position_before_lift,
    )

    print(
        "Object position after lift:",
        object_position_after_lift,
    )

    object_delta = (
        object_position_after_lift
        - object_position_before_lift
    )

    print(
        "Object displacement:",
        object_delta,
    )

    print(
        "Object vertical displacement:",
        object_delta[2],
    )

    left_actual_q = get_arm_qpos(
        left_qpos_ids
    )

    right_actual_q = get_arm_qpos(
        right_qpos_ids
    )


    print("\nLEFT ARM")

    print(
        "Commanded lift q:",
        left_lift_q,
    )

    print(
        "Actual q:",
        left_actual_q,
    )

    print(
        "q error norm:",
        np.linalg.norm(
            left_lift_q
            - left_actual_q
        ),
    )


    print("\nRIGHT ARM")

    print(
        "Commanded lift q:",
        right_lift_q,
    )

    print(
        "Actual q:",
        right_actual_q,
    )

    print(
        "q error norm:",
        np.linalg.norm(
            right_lift_q
            - right_actual_q
        ),
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Keep viewer open briefly
    # --------------------------------------------------------

    print(
        "\nHolding final lifted configuration..."
    )

    for _ in range(
        2500
    ):

        data.ctrl[
            left_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        data.ctrl[
            right_gripper_actuator_id
        ] = CLOSE_GRIPPER_CTRL

        step_with_arm_targets(
            left_lift_q,
            right_lift_q,
        )

        viewer.sync()

        time.sleep(
            0.001
        )
