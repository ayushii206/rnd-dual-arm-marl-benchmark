# Dual-Arm MARL Benchmark — Setup and Development Documentation

## 1. Project Overview

**Project title:**  
Benchmarking Classical Control and Decentralized Multi-Agent Reinforcement Learning for Cooperative Dual-Arm Object Transport

The objective of this project is to benchmark classical robotic control methods against decentralized multi-agent reinforcement learning methods for cooperative object transport using two robotic manipulators.

The project is simulation-only and uses MuJoCo with two Franka Emika Panda manipulators.

The planned methods are:

- Classical control
- Independent Proximal Policy Optimization (IPPO)
- Multi-Agent Proximal Policy Optimization (MAPPO)

All approaches will operate on the same cooperative transport task and will later be evaluated using common performance and robustness metrics.

---

## 2. Planned Evaluation Metrics

The main evaluation metrics are:

- Task success rate
- Task completion time
- Object stability
- Collision count
- Control smoothness
- Training convergence for reinforcement learning methods
- Robustness to environmental variations and disturbances

Planned robustness tests include:

- Sensor noise
- Action noise
- Object mass variation
- Friction variation
- External disturbances

---

## 3. Development Environment

Development is currently performed on:

- macOS Sequoia
- Intel-based MacBook Pro
- Python 3.11
- MuJoCo 3.8.1

A Python virtual environment is used for development.

Activate the environment with:

```bash
source .venv/bin/activate
```

---

## 4. MuJoCo Installation

The latest MuJoCo release could not be installed directly on the Intel macOS development machine because recent PyPI releases do not provide the required macOS x86_64 binary wheel.

MuJoCo 3.8.1 was therefore installed:

```bash
pip install mujoco==3.8.1
```

The installation was verified with:

```bash
python -c "import mujoco; print('MuJoCo installation successful')"
```

Expected output:

```text
MuJoCo installation successful
```

Viewer-based scripts on macOS are executed using:

```bash
mjpython <script>
```

rather than standard Python when required by the MuJoCo viewer.

---

## 5. Franka Emika Panda Model

The Franka Emika Panda model is based on the MuJoCo Menagerie Panda model.

The customized Panda model is stored under:

```text
assets/models/custom_panda/
```

A custom end-effector site was added to the Panda hand.

The modified model is:

```text
assets/models/custom_panda/panda_with_site.xml
```

The end-effector site is defined approximately as:

```xml
<site
    name="ee_site"
    pos="0 0 0.10"
    size="0.01"
    rgba="1 0 0 1"
/>
```

This site is used for:

- End-effector position tracking
- Jacobian calculation
- Inverse kinematics
- Future Cartesian control
- Future grasp pose definition

---

## 6. Single-Panda Validation

The Panda model was inspected to verify joints, actuators, bodies, and model dimensions.

The Panda contains the following arm joints:

```text
joint1
joint2
joint3
joint4
joint5
joint6
joint7
```

and two finger joints:

```text
finger_joint1
finger_joint2
```

The single Panda model has:

```text
nq = 9
nv = 9
nu = 8
```

The end-effector site was successfully detected and tracked.

Example initial end-effector position:

```text
[0.088, 0.000, 0.826]
```

Single-arm movement was successfully tested before constructing the dual-arm environment.

---

## 7. Dual-Panda Model Generation

The dual-arm environment is generated from the customized Panda model.

The generation script is:

```text
scripts/create_dual_panda_models.py
```

It creates:

```text
assets/models/dual_panda/left_panda.xml
assets/models/dual_panda/right_panda.xml
```

Names are prefixed with either:

```text
left_
```

or:

```text
right_
```

to prevent conflicts between the two robots.

Examples include:

```text
left_joint1
left_joint2
...
left_joint7

right_joint1
right_joint2
...
right_joint7
```

The end-effector sites become:

```text
left_ee_site
right_ee_site
```

Actuator names are also prefixed.

---

## 8. Dual-Arm Scene Construction

Directly including two complete Panda XML files caused duplicate MuJoCo default-class definitions.

Therefore, the combined scene is generated programmatically using:

```text
scripts/build_dual_arm_scene.py
```

The script uses one Panda model as the base and merges the required components of the second Panda into the same MuJoCo model.

The resulting scene is:

```text
assets/models/dual_panda/dual_arm_scene.xml
```

The two robots are placed on opposite sides of the central workspace.

### Left Panda placement

```python
child.set("pos", "0 0.65 0")
child.set("quat", "0.7071068 0 0 -0.7071068")
```

### Right Panda placement

```python
new_child.set("pos", "0 -0.65 0")
new_child.set("quat", "0.7071068 0 0 0.7071068")
```

The two robots therefore face inward toward the shared workspace.

The combined model has:

```text
nq = 18
nv = 18
nu = 16
```

---

## 9. Dual-Arm Control Validation

Independent actuator control was tested before introducing the transport task.

Relevant scripts include:

```text
scripts/view_dual_arm_scene.py
scripts/inspect_dual_arm_scene.py
scripts/test_dual_arm_control.py
```

The test confirmed that:

- The left Panda can move independently.
- The right Panda can remain stationary while the left Panda moves.
- The right Panda can move independently.
- The robots are correctly mirrored around the central workspace.
- Both robots expose independently accessible end-effector states.

Example inward end-effector positions reached during early testing were approximately:

```text
Left:
[0.0, 0.264, 0.548]

Right:
[0.0, -0.264, 0.548]
```

---

## 10. Cooperative Transport Scene

The cooperative transport scene is generated using:

```text
scripts/build_transport_scene.py
```

The generated model is:

```text
assets/models/dual_panda/transport_scene.xml
```

The scene contains:

- Two Panda manipulators
- A physical work table
- A dynamic transport object
- A transport-object center site
- Left and right grasp reference sites
- A goal region
- A goal-center site

---

## 11. Work Table

A physical work surface was added to prevent the transport object from falling to the floor.

The table is approximately defined as:

```xml
<geom
    name="work_table"
    type="box"
    pos="0.20 0 0.45"
    size="0.90 0.42 0.05"
    friction="1.0 0.005 0.0001"
/>
```

The table top is therefore at approximately:

```text
z = 0.50 m
```

---

## 12. Transport Object

The transport object is currently represented as a dynamic box.

Approximate parameters:

```text
Position:
[0.0, 0.0, 0.56]

Half-size:
[0.20, 0.12, 0.06]

Mass:
2.0 kg

Friction:
1.0 0.005 0.0001
```

The object uses a free joint:

```text
object_freejoint
```

This gives the object six physical degrees of freedom.

The complete transport model therefore has approximately:

```text
nq = 25
nv = 24
nu = 16
```

---

## 13. Object and Goal Sites

The transport object contains the following reference sites:

```text
object_center
left_grasp_site
right_grasp_site
```

Current approximate grasp-site positions in world coordinates at initialization are:

```text
left_grasp_site:
[0.0, 0.13, 0.56]

right_grasp_site:
[0.0, -0.13, 0.56]
```

The goal is located in the positive X direction.

Approximate goal center:

```text
[0.45, 0.0, 0.56]
```

The current grasp sites are reference points only. Final grasp geometry and grasp handles will be designed in a later development step.

---

## 14. Object Stability Validation

Object settling was tested using:

```text
scripts/test_object_settling.py
```

The object settles at approximately:

```text
z = 0.5599 m
```

which is consistent with the intended center height of:

```text
z = 0.56 m
```

The final object velocities after settling were approximately on the order of:

```text
1e-14 to 1e-18
```

and can therefore be considered numerically zero.

The initial transport-object placement is stable.

When inspecting MuJoCo derived quantities immediately after creating `MjData`, `mujoco.mj_forward()` must be called before reading derived positions.

---

## 15. Initial Robot Configuration for IK

Using the all-zero Panda joint configuration caused the first position-IK test to approach joint limits.

In particular, Panda joints 4 and 6 were close to their limits.

A bent nominal configuration was therefore selected:

```python
initial_q = np.array([
    0.0,
    -0.4,
    0.0,
    -1.8,
    0.0,
    1.4,
    0.8,
])
```

This configuration provides substantially better workspace for iterative inverse kinematics.

---

## 16. Position-Only Inverse Kinematics

Position-only inverse kinematics was implemented using damped least squares.

The target is an XYZ end-effector position.

The update is based on the translational end-effector Jacobian.

Conceptually:

```text
current end-effector position
        |
        v
position error
        |
        v
translational Jacobian
        |
        v
damped least-squares solution
        |
        v
joint update
        |
        v
forward kinematics
        |
        v
repeat until tolerance
```

The primary IK parameters used during validation were:

```text
Maximum iterations: 2000
Position tolerance: 1e-4 m
Damping: 0.05
Step size: 0.2
```

---

## 17. Left-Arm Position IK

The initial left-arm approach target was defined as:

```text
[0.00, 0.21, 0.56]
```

This position lies outside the current left grasp reference point and acts as a safe approach location.

The validated result was approximately:

```text
Final EE position:
[0.000000, 0.210063, 0.560056]

Final error:
8.466e-05 m
```

This corresponds to approximately:

```text
0.085 mm
```

of Cartesian position error.

The solver converged after approximately:

```text
29 iterations
```

The corresponding joint solution was approximately:

```text
joint1 =  0.00000
joint2 = -0.30607
joint3 =  0.00000
joint4 = -1.80148
joint5 =  0.00000
joint6 =  1.45606
joint7 =  0.80000
```

No joint was at its limit.

---

## 18. Right-Arm Position IK

The mirrored right-arm target was:

```text
[0.00, -0.21, 0.56]
```

The validated result was approximately:

```text
Final EE position:
[0.000000, -0.210063, 0.560056]

Final error:
8.466e-05 m
```

The solver again converged after approximately:

```text
29 iterations
```

Because the two Panda bases are mirrored in the world, the local joint solution is effectively identical to the left-arm solution.

This validates the geometric symmetry of the dual-arm environment.

---

## 19. Dynamic Actuator Tracking

Numerical IK modifies generalized coordinates directly and does not by itself demonstrate that the simulated actuators can physically track the resulting configuration.

Therefore, the IK configurations were separately tested using the Panda actuators.

Relevant scripts include:

```text
scripts/move_left_arm_to_approach.py
scripts/move_right_arm_to_approach.py
scripts/move_both_arms_to_approach.py
```

The tests confirmed:

- Left-arm actuator tracking
- Right-arm actuator tracking
- Stationary opposite-arm behavior
- Simultaneous dual-arm motion
- Stable mirrored movement toward the transport object

During simultaneous motion, the final end-effector positions were approximately:

```text
Left:
[-0.00003, 0.20998, 0.55533]

Right:
[0.00003, -0.20998, 0.55533]
```

The small difference between the mathematical IK solution and dynamically tracked position is expected because the dynamic simulation uses physical actuators rather than direct `qpos` assignment.

---

## 20. Reusable IK Controller

The position IK implementation was refactored into:

```text
src/controllers/ik_controller.py
```

The reusable position solver is:

```python
solve_position_ik(...)
```

It accepts:

- MuJoCo model
- MuJoCo data
- Joint names
- End-effector site name
- Cartesian target position
- Initial joint configuration
- IK solver parameters

It returns:

- Joint-space IK solution
- Final Cartesian position
- Final Cartesian error

The reusable solver was validated simultaneously for both robots using:

```text
scripts/test_dual_arm_ik.py
```

The reusable implementation reproduced the previous position-IK results.

---

## 21. End-Effector Orientation Inspection

The end-effector rotation matrices were inspected using:

```text
scripts/inspect_ee_orientation.py
```

At the nominal initial configuration, the left end-effector orientation was approximately:

```text
[[-0.0146, -0.9999,  0.0000],
 [-0.9999,  0.0146,  0.0000],
 [ 0.0000,  0.0000, -1.0000]]
```

The right end-effector orientation was approximately:

```text
[[ 0.0146,  0.9999,  0.0000],
 [ 0.9999, -0.0146,  0.0000],
 [ 0.0000,  0.0000, -1.0000]]
```

The two orientations are approximately mirrored, consistent with the mirrored robot-base configuration.

---

## 22. Six-Dimensional Pose IK

The reusable IK controller was extended with:

```python
solve_pose_ik(...)
```

This solver controls both:

- End-effector XYZ position
- End-effector orientation

The pose solver uses both components of the MuJoCo site Jacobian:

```text
jacp
```

for translation and:

```text
jacr
```

for rotation.

The combined Jacobian therefore represents a six-dimensional Cartesian pose objective.

The Panda has seven arm joints, providing one redundant joint degree of freedom relative to the six-dimensional Cartesian task.

---

## 23. Pose-IK Validation

The first pose-IK validation deliberately preserved each robot's nominal end-effector orientation while moving the end effector to its approach position.

This separated validation of the pose solver from later grasp-orientation design.

The test script is:

```text
scripts/test_dual_arm_pose_ik.py
```

### Left Arm

Target:

```text
Position:
[0.00, 0.21, 0.56]

Orientation:
nominal left end-effector orientation
```

Final position:

```text
[0.000000, 0.210066, 0.560073]
```

Position error:

```text
9.855e-05 m
```

Approximate joint solution:

```text
joint1 =  0.00000
joint2 = -0.33368
joint3 =  0.00000
joint4 = -1.83626
joint5 =  0.00000
joint6 =  1.50257
joint7 =  0.80000
```

### Right Arm

Target:

```text
Position:
[0.00, -0.21, 0.56]

Orientation:
nominal right end-effector orientation
```

Final position:

```text
[0.000000, -0.210066, 0.560073]
```

Position error:

```text
9.855e-05 m
```

The right-arm joint solution was effectively identical to the left-arm solution because of the symmetric robot placement.

The remaining rotation-matrix differences were approximately on the order of:

```text
1e-05
```

demonstrating successful orientation preservation during Cartesian movement.

---

## 24. Current Validation Status

The following components have been validated:

- [x] MuJoCo installation
- [x] Single Panda model
- [x] Custom end-effector site
- [x] Single-arm movement
- [x] Dual-Panda model generation
- [x] Mirrored dual-arm placement
- [x] Independent left/right actuator control
- [x] Dynamic transport object
- [x] Stable work surface
- [x] Stable initial object placement
- [x] Object state access
- [x] Object grasp reference sites
- [x] Goal region
- [x] Left position IK
- [x] Right position IK
- [x] Joint-limit handling
- [x] Left dynamic IK tracking
- [x] Right dynamic IK tracking
- [x] Simultaneous dual-arm approach
- [x] Reusable position IK controller
- [x] End-effector orientation inspection
- [x] Six-dimensional pose IK
- [x] Mirrored dual-arm pose-IK validation

---

## 25. Current Development Boundary

Development is currently complete through:

```text
Dual-arm 6D pose IK validation
```

The current pipeline is:

```text
MuJoCo setup
    |
    v
Single Panda validation
    |
    v
Dual-Panda environment
    |
    v
Transport task scene
    |
    v
Independent actuator control
    |
    v
Position IK
    |
    v
Dynamic IK tracking
    |
    v
Simultaneous dual-arm approach
    |
    v
Reusable IK controller
    |
    v
6D pose IK
    |
    v
GRASP DESIGN  <-- CURRENT NEXT STEP
```

---

## 26. Next Development Steps

The next development phase is cooperative grasping.

The planned order is:

1. Design grasp geometry suitable for the Panda grippers.
2. Add grasp handles or graspable regions to opposite sides of the transport object.
3. Position the left and right grasp sites relative to the new geometry.
4. Determine the required left and right gripper orientations.
5. Define safe pre-grasp poses.
6. Solve 6D IK for the pre-grasp poses.
7. Dynamically move both arms to pre-grasp.
8. Move from pre-grasp to grasp.
9. Control the Panda fingers.
10. Validate stable dual-arm object contact.
11. Lift or translate the object cooperatively.
12. Develop the complete classical cooperative transport baseline.

After the classical baseline is functional, development will proceed to:

```text
Gymnasium/MARL environment
        |
        v
Observation and action spaces
        |
        v
Reward design
        |
        v
IPPO
        |
        v
MAPPO
        |
        v
Benchmark experiments
        |
        v
Robustness experiments
        |
        v
Analysis
```

---

## 27. Important Design Notes

### IK Initialization

Do not use the all-zero Panda configuration as the default IK initialization.

Use:

```python
initial_q = np.array([
    0.0,
    -0.4,
    0.0,
    -1.8,
    0.0,
    1.4,
    0.8,
])
```

unless a later task-specific nominal configuration replaces it.

### Position vs Pose IK

`solve_position_ik()` should remain available for tasks requiring only Cartesian position control.

`solve_pose_ik()` should be used when both end-effector position and orientation matter, particularly during grasping.

### Grasp Orientation

The currently validated pose IK preserves the nominal wrist orientation.

This should not automatically be treated as the final grasp orientation.

The final grasp orientation must be selected based on:

- Panda finger opening direction
- Handle geometry
- Object geometry
- Collision avoidance
- Stable force/contact configuration

### Test Scripts

Successful validation scripts should be preserved rather than overwritten.

They provide regression tests and evidence of incremental development.

---

## 28. Current Project Structure

Relevant files currently include approximately:

```text
rnd-dual-arm-marl-benchmark/
|
|-- assets/
|   `-- models/
|       |-- custom_panda/
|       |   |-- panda_with_site.xml
|       |   `-- assets/
|       |
|       `-- dual_panda/
|           |-- left_panda.xml
|           |-- right_panda.xml
|           |-- dual_arm_scene.xml
|           `-- transport_scene.xml
|
|-- docs/
|   `-- setup.md
|
|-- scripts/
|   |-- create_dual_panda_models.py
|   |-- build_dual_arm_scene.py
|   |-- build_transport_scene.py
|   |-- view_dual_arm_scene.py
|   |-- inspect_dual_arm_scene.py
|   |-- test_dual_arm_control.py
|   |-- view_transport_scene.py
|   |-- inspect_transport_scene.py
|   |-- test_object_settling.py
|   |-- test_left_arm_ik.py
|   |-- test_right_arm_ik.py
|   |-- move_left_arm_to_approach.py
|   |-- move_right_arm_to_approach.py
|   |-- move_both_arms_to_approach.py
|   |-- test_dual_arm_ik.py
|   |-- inspect_ee_orientation.py
|   `-- test_dual_arm_pose_ik.py
|
`-- src/
    `-- controllers/
        `-- ik_controller.py
```

---

## 29. Reproducibility

Before running scripts, activate the virtual environment:

```bash
source .venv/bin/activate
```

Non-viewer scripts can normally be run using:

```bash
python scripts/<script_name>.py
```

Viewer scripts on macOS should be run using:

```bash
mjpython scripts/<script_name>.py
```

The transport scene can be regenerated using:

```bash
python scripts/build_transport_scene.py
```

The generated XML files should remain reproducible from the corresponding generation scripts rather than being edited manually wherever possible.

---

## 30. Current Checkpoint

As of the current development checkpoint, both Panda robots can:

- Exist simultaneously in the same MuJoCo environment.
- Be actuated independently.
- Move simultaneously.
- Access their individual end-effector states.
- Solve position-only inverse kinematics.
- Solve full position-and-orientation inverse kinematics.
- Reach mirrored approach positions around the transport object.
- Preserve mirrored end-effector orientations during pose IK.

The next technical milestone is:

> **Design and validate a physically meaningful dual-arm grasp of the transport object.**
