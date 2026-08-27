import time
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_PATH = Path(
    "assets/models/mujoco_menagerie/franka_emika_panda/scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

target = data.ctrl.copy()

target[:7] = [
    0.0,
    -0.5,
    0.0,
    -2.0,
    0.0,
    1.5,
    0.8
]

step = 0

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        data.ctrl[:] = target

        mujoco.mj_step(model, data)

        # Print approximately every 500 simulation steps
        if step % 500 == 0:

            hand_position = data.xpos[hand_id].copy()

            print(
                f"Step {step:6d} | "
                f"Hand xyz = "
                f"{hand_position[0]: .3f}, "
                f"{hand_position[1]: .3f}, "
                f"{hand_position[2]: .3f}"
            )

        viewer.sync()

        time.sleep(model.opt.timestep)

        step += 1
