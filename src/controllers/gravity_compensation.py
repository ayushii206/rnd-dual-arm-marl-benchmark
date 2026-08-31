import mujoco


def get_joint_dof_ids(
    model,
    actuator_ids,
):
    """
    Return the MuJoCo DOF indices associated with a list
    of single-joint actuators.
    """

    dof_ids = []

    for actuator_id in actuator_ids:

        joint_id = model.actuator_trnid[
            actuator_id,
            0,
        ]

        dof_id = model.jnt_dofadr[
            joint_id
        ]

        dof_ids.append(
            dof_id
        )

    return dof_ids


def apply_bias_compensation(
    model,
    data,
    dof_ids,
):
    """
    Apply MuJoCo generalized bias forces as feed-forward
    compensation to the selected robot DOFs.

    Call this before mj_step().
    """

    for dof_id in dof_ids:

        data.qfrc_applied[
            dof_id
        ] = data.qfrc_bias[
            dof_id
        ]
