from pathlib import Path

import mujoco

MODEL_PATH = Path(
    "assets/models/dual_panda/transport_scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

object_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "transport_object",
)

print("Initial object position:")
print(data.xpos[object_id])

for _ in range(2000):
    mujoco.mj_step(model, data)

print("\nSettled object position:")
print(data.xpos[object_id])

print("\nObject velocity:")
print(data.qvel[-6:])

