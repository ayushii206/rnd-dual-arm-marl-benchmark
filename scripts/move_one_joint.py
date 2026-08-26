import time
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_PATH = Path(
    "assets/models/mujoco_menagerie/franka_emika_panda/scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("Number of actuators:", model.nu)

# Start from current control values.
target = data.ctrl.copy()

# Move only actuator 0 slightly.
target[0] = 0.5

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        data.ctrl[:] = target

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)
