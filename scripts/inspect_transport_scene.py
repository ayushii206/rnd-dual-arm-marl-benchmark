from pathlib import Path
import mujoco

MODEL_PATH = Path(
    "assets/models/dual_panda/transport_scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

site_names = [
    "left_ee_site",
    "right_ee_site",
    "object_center",
    "left_grasp_site",
    "right_grasp_site",
    "goal_center",
]

for name in site_names:

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        name,
    )

    print(
        f"{name:18s} "
        f"id={site_id:2d} "
        f"pos={data.site_xpos[site_id]}"
    )

object_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "transport_object",
)

print("\nObject body ID:", object_id)
print("Object position:", data.xpos[object_id])
print("Object orientation matrix:")
print(data.xmat[object_id].reshape(3, 3))
