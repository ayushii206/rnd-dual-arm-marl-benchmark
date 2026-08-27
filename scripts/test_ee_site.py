from pathlib import Path
import time
import mujoco
import mujoco.viewer

MODEL_PATH = Path(
    "assets/models/custom_panda/scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "ee_site"
)

print("ee_site ID:", site_id)
print("ee_site position:", data.site_xpos[site_id])

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

