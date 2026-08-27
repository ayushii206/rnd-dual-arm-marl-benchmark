from pathlib import Path
import mujoco

MODEL_PATH = Path(
    "assets/models/mujoco_menagerie/franka_emika_panda/scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# Compute initial state
mujoco.mj_forward(model, data)

print("\n=== MODEL DIMENSIONS ===")
print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)

print("\n=== JOINT POSITIONS qpos ===")
print(data.qpos)

print("\n=== JOINT VELOCITIES qvel ===")
print(data.qvel)

print("\n=== CONTROL INPUTS ctrl ===")
print(data.ctrl)

# Find the hand body
hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

print("\n=== END EFFECTOR ===")
print("Hand body ID:", hand_id)
print("Hand position [x, y, z]:")
print(data.xpos[hand_id])

print("\nHand rotation matrix:")
print(data.xmat[hand_id].reshape(3, 3))

print("\n=== JOINT STATES ===")

for i in range(model.njnt):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    qpos_address = model.jnt_qposadr[i]

    print(
        f"{name:15s} "
        f"qpos = {data.qpos[qpos_address]: .4f}"
    )
