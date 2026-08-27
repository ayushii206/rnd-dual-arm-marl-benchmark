from pathlib import Path
import mujoco


models = [
    Path("assets/models/dual_panda/left_panda.xml"),
    Path("assets/models/dual_panda/right_panda.xml"),
]

for path in models:

    print(f"\nTesting {path}")

    try:
        model = mujoco.MjModel.from_xml_path(str(path))

        print("Loaded successfully")
        print("nq:", model.nq)
        print("nv:", model.nv)
        print("nu:", model.nu)

    except Exception as error:

        print("FAILED")
        print(error)

