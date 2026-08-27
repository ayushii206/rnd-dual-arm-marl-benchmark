import time
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_PATH = Path(
    "assets/models/dual_panda/dual_arm_scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)

