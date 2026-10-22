
import numpy as np
import mujoco

from src.envs.dual_arm_transport_env import ACTION_SCALE_RAD


DAMPING = 0.05
MAX_ACTION_MAGNITUDE = 0.25


def calculate_cartesian_action(
    env,
    site_id,
    dof_ids,
    desired_displacement,
):
    """
    Convert a desired 3D end-effector displacement
    into a normalized 7D joint action.

    Uses damped least-squares inverse kinematics.
    """

    desired_displacement = np.asarray(
        desired_displacement,
        dtype=np.float64,
    )

    if desired_displacement.shape != (3,):
        raise ValueError(
            "desired_displacement must have shape (3,)."
        )

    if not np.all(np.isfinite(desired_displacement)):
        raise ValueError(
            "desired_displacement must be finite."
        )

    nv = env.model.nv

    jac_pos = np.zeros((3, nv))
    jac_rot = np.zeros((3, nv))

    mujoco.mj_jacSite(
        env.model,
        env.data,
        jac_pos,
        jac_rot,
        site_id,
    )

    position_jacobian = jac_pos[:, dof_ids]
    rotation_jacobian = jac_rot[:, dof_ids]

    jacobian = np.vstack([
        position_jacobian,
        0.5 * rotation_jacobian,
    ])

    desired_motion = np.concatenate([
        desired_displacement,
        np.zeros(3),
    ])

    regularized = (
        jacobian @ jacobian.T
        + (DAMPING ** 2) * np.eye(6)
    )

    joint_delta = jacobian.T @ np.linalg.solve(
        regularized,
        desired_motion,
    )

    action = joint_delta / ACTION_SCALE_RAD

    return np.clip(
        action,
        -MAX_ACTION_MAGNITUDE,
        MAX_ACTION_MAGNITUDE,
    )

