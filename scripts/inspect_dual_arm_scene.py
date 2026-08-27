import mujoco
from pathlib import Path

MODEL_PATH = Path(
    "assets/models/dual_panda/dual_arm_scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

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

print("left_ee_site ID:", left_site_id)
print("right_ee_site ID:", right_site_id)

print(
    "left_ee_site position:",
    data.site_xpos[left_site_id]
)

print(
    "right_ee_site position:",
    data.site_xpos[right_site_id]
)

print("\nActuators:")

for i in range(model.nu):
    print(
        i,
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i
        )
    )
