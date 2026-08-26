import mujoco
from pathlib import Path

MODEL_PATH = Path(
    "assets/models/mujoco_menagerie/franka_emika_panda/scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

print("\nJOINTS")
print("-" * 40)

for i in range(model.njnt):
    print(
        i,
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            i
        )
    )

print("\nACTUATORS")
print("-" * 40)

for i in range(model.nu):
    print(
        i,
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i
        )
    )

print("\nBODIES")
print("-" * 40)

for i in range(model.nbody):
    print(
        i,
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            i
        )
    )

print("\nSITES")
print("-" * 40)

for i in range(model.nsite):
    print(
        i,
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            i
        )
    )
