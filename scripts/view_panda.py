import mujoco
import mujoco.viewer
import time
from pathlib import Path

MODEL_PATH = Path(
    "assets/models/mujoco_menagerie/franka_emika_panda/scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("Panda model loaded successfully")
print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
