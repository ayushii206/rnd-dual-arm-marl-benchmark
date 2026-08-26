import time
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_PATH = Path(
    "assets/models/mujoco_menagerie/franka_emika_panda/scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# Target joint positions for the 7 arm actuators
target = data.ctrl.copy()

target[0] = 0.0
target[1] = -0.5
target[2] = 0.0
target[3] = -2.0
target[4] = 0.0
target[5] = 1.5
target[6] = 0.8

# Keep gripper actuator unchanged
# target[7] stays at its default

print("Applying target control:", target)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        data.ctrl[:] = target

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)
