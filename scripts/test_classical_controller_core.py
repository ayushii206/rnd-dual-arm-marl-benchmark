from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.controllers.classical_controller import (
    ClassicalController,
    ClassicalPhase,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "assets/models/dual_panda/transport_scene.xml"
)


INITIAL_ARM_Q = np.array([
    0.0,
    -0.4,
    0.0,
    -1.8,
    0.0,
    1.4,
    0.8,
])


print(
    "=" * 70
)

print(
    "CLASSICAL CONTROLLER CORE SMOKE TEST"
)

print(
    "=" * 70
)


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    model
)


controller = ClassicalController(
    model,
    data,
)


print(
    "Model nq:",
    model.nq,
)

print(
    "Model nv:",
    model.nv,
)

print(
    "Model nu:",
    model.nu,
)


print(
    "\nLeft qpos IDs:",
    controller.left_qpos_ids,
)

print(
    "Right qpos IDs:",
    controller.right_qpos_ids,
)


print(
    "\nLeft arm actuator IDs:",
    controller.left_arm_actuator_ids,
)

print(
    "Right arm actuator IDs:",
    controller.right_arm_actuator_ids,
)


print(
    "\nLeft gripper actuator ID:",
    controller.left_gripper_actuator_id,
)

print(
    "Right gripper actuator ID:",
    controller.right_gripper_actuator_id,
)


if len(
    controller.left_arm_actuator_ids
) != 7:
    raise RuntimeError(
        "Left arm actuator count is not 7."
    )


if len(
    controller.right_arm_actuator_ids
) != 7:
    raise RuntimeError(
        "Right arm actuator count is not 7."
    )


controller.set_arm_qpos(
    INITIAL_ARM_Q,
    INITIAL_ARM_Q,
)


controller.open_grippers()


mujoco.mj_forward(
    model,
    data,
)


left_before = (
    controller.get_left_arm_qpos()
)

right_before = (
    controller.get_right_arm_qpos()
)


print(
    "\nLeft arm q before step:",
    left_before,
)

print(
    "Right arm q before step:",
    right_before,
)


# Hold exactly the current pose for a few physics steps.
#
# This is intentionally NOT a manipulation experiment.
# It only verifies the reusable low-level controller API.

for _ in range(
    10
):
    controller.step(
        INITIAL_ARM_Q,
        INITIAL_ARM_Q,
    )


left_after = (
    controller.get_left_arm_qpos()
)

right_after = (
    controller.get_right_arm_qpos()
)


print(
    "\nLeft arm q after 10 steps:",
    left_after,
)

print(
    "Right arm q after 10 steps:",
    right_after,
)


left_error = float(
    np.linalg.norm(
        left_after
        - INITIAL_ARM_Q
    )
)

right_error = float(
    np.linalg.norm(
        right_after
        - INITIAL_ARM_Q
    )
)


print(
    "\nLeft pose error norm:",
    left_error,
)

print(
    "Right pose error norm:",
    right_error,
)


controller.set_phase(
    ClassicalPhase.GRASP
)


print(
    "\nController phase:",
    controller.get_phase().name
)


print(
    "\nCLASSICAL CONTROLLER CORE: PASS"
)

print(
    "=" * 70
)
