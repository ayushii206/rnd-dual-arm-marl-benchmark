import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = Path(
    "assets/models/dual_panda/dual_arm_scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# IDs of end-effector sites
left_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "left_ee_site"
)

right_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_ee_site"
)

# Neutral-ish target poses
left_target = np.array([
    0.0,
    -0.5,
    0.0,
    -2.0,
    0.0,
    1.5,
    0.8
])

right_target = np.array([
    0.0,
    -0.5,
    0.0,
    -2.0,
    0.0,
    1.5,
    0.8
])

step = 0

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # Phase 1: keep everything at zero
        if step < 1000:
            data.ctrl[:] = 0.0

        # Phase 2: move only left arm
        elif step < 3000:
            data.ctrl[0:7] = left_target
            data.ctrl[8:15] = 0.0

        # Phase 3: move both arms
        else:
            data.ctrl[0:7] = left_target
            data.ctrl[8:15] = right_target

        mujoco.mj_step(model, data)

        if step % 500 == 0:

            left_pos = data.site_xpos[left_site_id].copy()
            right_pos = data.site_xpos[right_site_id].copy()

            print(
                f"Step {step:5d} | "
                f"Left EE: {left_pos} | "
                f"Right EE: {right_pos}"
            )

        viewer.sync()

        time.sleep(model.opt.timestep)

        step += 1
