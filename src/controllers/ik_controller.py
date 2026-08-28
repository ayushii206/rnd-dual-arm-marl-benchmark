import mujoco
import numpy as np


def solve_position_ik(
    model,
    data,
    joint_names,
    site_name,
    target_position,
    initial_q,
    max_iterations=2000,
    tolerance=1e-4,
    damping=0.05,
    step_size=0.2,
):
    """
    Solve position-only inverse kinematics for one robot arm.

    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.

    data : mujoco.MjData
        MuJoCo data object.

    joint_names : list[str]
        Names of the controlled arm joints.

    site_name : str
        End-effector site name.

    target_position : np.ndarray
        Desired XYZ position in world coordinates.

    initial_q : np.ndarray
        Initial joint configuration.

    Returns
    -------
    solution : np.ndarray
        Solved joint configuration.

    final_position : np.ndarray
        Final end-effector XYZ position.

    final_error : float
        Euclidean position error.
    """

    joint_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        for name in joint_names
    ]

    qpos_ids = [
        model.jnt_qposadr[joint_id]
        for joint_id in joint_ids
    ]

    dof_ids = [
        model.jnt_dofadr[joint_id]
        for joint_id in joint_ids
    ]

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id == -1:
        raise RuntimeError(
            f"Site not found: {site_name}"
        )

    for qpos_id, value in zip(
        qpos_ids,
        initial_q,
    ):
        data.qpos[qpos_id] = value

    mujoco.mj_forward(model, data)

    for iteration in range(max_iterations):

        current_position = (
            data.site_xpos[site_id].copy()
        )

        error = (
            target_position
            - current_position
        )

        error_norm = np.linalg.norm(error)

        if error_norm < tolerance:
            break

        jacp = np.zeros(
            (3, model.nv)
        )

        jacr = np.zeros(
            (3, model.nv)
        )

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            site_id,
        )

        J = jacp[:, dof_ids]

        A = (
            J @ J.T
            + (damping ** 2)
            * np.eye(3)
        )

        dq = (
            J.T
            @ np.linalg.solve(
                A,
                step_size * error,
            )
        )

        for i, qpos_id in enumerate(
            qpos_ids
        ):

            data.qpos[qpos_id] += dq[i]

            joint_id = joint_ids[i]

            if model.jnt_limited[
                joint_id
            ]:

                lower = (
                    model.jnt_range[
                        joint_id, 0
                    ]
                )

                upper = (
                    model.jnt_range[
                        joint_id, 1
                    ]
                )

                data.qpos[qpos_id] = (
                    np.clip(
                        data.qpos[
                            qpos_id
                        ],
                        lower,
                        upper,
                    )
                )

        mujoco.mj_forward(
            model,
            data,
        )

    final_position = (
        data.site_xpos[site_id].copy()
    )

    final_error = np.linalg.norm(
        target_position
        - final_position
    )

    solution = np.array(
        [
            data.qpos[qpos_id]
            for qpos_id
            in qpos_ids
        ]
    )

    return (
        solution,
        final_position,
        final_error,
    )

def solve_pose_ik(
    model,
    data,
    joint_names,
    site_name,
    target_position,
    target_rotation,
    initial_q,
    max_iterations=3000,
    position_tolerance=1e-4,
    orientation_tolerance=1e-3,
    damping=0.05,
    step_size=0.15,
    orientation_weight=0.5,
):
    """
    Solve 6D pose IK:
    position + orientation.
    """

    joint_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        for name in joint_names
    ]

    qpos_ids = [
        model.jnt_qposadr[joint_id]
        for joint_id in joint_ids
    ]

    dof_ids = [
        model.jnt_dofadr[joint_id]
        for joint_id in joint_ids
    ]

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id == -1:
        raise RuntimeError(
            f"Site not found: {site_name}"
        )

    for qpos_id, value in zip(
        qpos_ids,
        initial_q,
    ):
        data.qpos[qpos_id] = value

    mujoco.mj_forward(model, data)

    for _ in range(max_iterations):

        current_position = (
            data.site_xpos[site_id].copy()
        )

        current_rotation = (
            data.site_xmat[site_id]
            .reshape(3, 3)
            .copy()
        )

        position_error = (
            target_position
            - current_position
        )

        rotation_error_matrix = (
            target_rotation
            @ current_rotation.T
        )

        orientation_error = np.array([
            rotation_error_matrix[2, 1]
            - rotation_error_matrix[1, 2],

            rotation_error_matrix[0, 2]
            - rotation_error_matrix[2, 0],

            rotation_error_matrix[1, 0]
            - rotation_error_matrix[0, 1],
        ]) * 0.5

        position_error_norm = np.linalg.norm(
            position_error
        )

        orientation_error_norm = np.linalg.norm(
            orientation_error
        )

        if (
            position_error_norm
            < position_tolerance
            and orientation_error_norm
            < orientation_tolerance
        ):
            break

        jacp = np.zeros(
            (3, model.nv)
        )

        jacr = np.zeros(
            (3, model.nv)
        )

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            site_id,
        )

        J_position = jacp[:, dof_ids]
        J_rotation = jacr[:, dof_ids]

        J = np.vstack([
            J_position,
            orientation_weight * J_rotation,
        ])

        error = np.concatenate([
            position_error,
            orientation_weight
            * orientation_error,
        ])

        A = (
            J @ J.T
            + (damping ** 2)
            * np.eye(6)
        )

        dq = (
            J.T
            @ np.linalg.solve(
                A,
                step_size * error,
            )
        )

        for i, qpos_id in enumerate(
            qpos_ids
        ):

            data.qpos[qpos_id] += dq[i]

            joint_id = joint_ids[i]

            if model.jnt_limited[
                joint_id
            ]:

                lower = (
                    model.jnt_range[
                        joint_id, 0
                    ]
                )

                upper = (
                    model.jnt_range[
                        joint_id, 1
                    ]
                )

                data.qpos[qpos_id] = np.clip(
                    data.qpos[qpos_id],
                    lower,
                    upper,
                )

        mujoco.mj_forward(
            model,
            data,
        )

    final_position = (
        data.site_xpos[site_id].copy()
    )

    final_rotation = (
        data.site_xmat[site_id]
        .reshape(3, 3)
        .copy()
    )

    solution = np.array([
        data.qpos[qpos_id]
        for qpos_id in qpos_ids
    ])

    return (
        solution,
        final_position,
        final_rotation,
    )
