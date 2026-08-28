from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = Path(
    "assets/models/dual_panda/transport_scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)


# ------------------------------------------------------------
# Right Panda joint names
# ------------------------------------------------------------

joint_names = [
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7",
]

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

# ------------------------------------------------------------
# Set a reasonable initial configuration
# ------------------------------------------------------------

initial_q = np.array([
    0.0,      # joint1
    -0.4,     # joint2
    0.0,      # joint3
    -1.8,     # joint4
    0.0,      # joint5
    1.4,      # joint6
    0.8,      # joint7
])

for qpos_id, value in zip(qpos_ids, initial_q):
    data.qpos[qpos_id] = value

mujoco.mj_forward(model, data)

# ------------------------------------------------------------
# right end-effector site
# ------------------------------------------------------------

site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_ee_site",
)


# ------------------------------------------------------------
# Safe approach target near the right side of the box
# ------------------------------------------------------------

target_position = np.array([
    0.00,
    -0.21,
    0.56,
])


# ------------------------------------------------------------
# Damped least-squares IK parameters
# ------------------------------------------------------------

max_iterations = 2000
tolerance = 1e-4
damping = 0.05
step_size = 0.2


print("Initial EE position:")
print(data.site_xpos[site_id])

print("\nTarget position:")
print(target_position)


# ------------------------------------------------------------
# IK loop
# ------------------------------------------------------------

for iteration in range(max_iterations):

    mujoco.mj_forward(model, data)

    current_position = data.site_xpos[site_id].copy()

    error = target_position - current_position

    error_norm = np.linalg.norm(error)

    if error_norm < tolerance:
        print(
            f"\nConverged after {iteration} iterations."
        )
        break

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(
        model,
        data,
        jacp,
        jacr,
        site_id,
    )

    # Use only right-arm DoFs
    J = jacp[:, dof_ids]

    # Damped least-squares update
    A = J @ J.T + (damping ** 2) * np.eye(3)

    dq = J.T @ np.linalg.solve(
        A,
        step_size * error,
    )

    # Update only right arm joint positions
    for i, qpos_id in enumerate(qpos_ids):

        data.qpos[qpos_id] += dq[i]

        joint_id = joint_ids[i]

        if model.jnt_limited[joint_id]:

            lower = model.jnt_range[joint_id, 0]
            upper = model.jnt_range[joint_id, 1]

            data.qpos[qpos_id] = np.clip(
                data.qpos[qpos_id],
                lower,
                upper,
            )

else:
    print("\nIK did not reach tolerance.")


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

mujoco.mj_forward(model, data)

final_position = data.site_xpos[site_id].copy()

print("\nFinal EE position:")
print(final_position)

print("\nPosition error:")
print(target_position - final_position)

print(
    "\nFinal error magnitude:",
    np.linalg.norm(
        target_position - final_position
    ),
)

print("\nSolved joint positions:")

for name, qpos_id in zip(
    joint_names,
    qpos_ids,
):
    print(
        f"{name:15s}: "
        f"{data.qpos[qpos_id]: .5f}"
    )

print("\nJoint limit check:")

for name, joint_id, qpos_id in zip(
    joint_names,
    joint_ids,
    qpos_ids,
):
    q = data.qpos[qpos_id]

    if model.jnt_limited[joint_id]:
        lower = model.jnt_range[joint_id, 0]
        upper = model.jnt_range[joint_id, 1]

        print(
            f"{name:15s}: "
            f"q={q: .5f}, "
            f"range=[{lower: .5f}, {upper: .5f}]"
        )
