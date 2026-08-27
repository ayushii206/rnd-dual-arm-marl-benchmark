# R&D Project — Setup, Installation, and Development Progress

## Project Title

**Benchmarking Classical Control and Decentralized Multi-Agent Reinforcement Learning for Cooperative Dual-Arm Object Transport**

## 1. Purpose of This Document

This document records the software environment, installation procedure, model setup, implementation steps, verification tests, and development progress of the R&D project.

The purpose is to ensure that:

- the development environment can be reproduced;
- important implementation decisions are documented;
- completed milestones can be tracked;
- experiments can later be reproduced on another computer or university compute cluster;
- the relationship between the MuJoCo implementation and the final benchmarking environment remains clear.

This document should be updated throughout the project as new components are implemented.

---

# 2. Project Objective

The objective of the project is to develop a simulation-based benchmarking framework for cooperative dual-arm object transport.

Two Franka Emika Panda robotic manipulators will ultimately cooperate to transport a shared rigid object toward a specified goal.

The final benchmark is intended to compare:

1. Classical cooperative control
2. Independent Proximal Policy Optimization (IPPO)
3. Multi-Agent Proximal Policy Optimization (MAPPO)

The methods will eventually be evaluated using common performance metrics and robustness conditions.

The current development phase focuses on constructing and validating the MuJoCo simulation environment before implementing the complete dual-arm task and learning algorithms.

---

# 3. Development Machine

Initial development is being performed on:

- Device: MacBook Pro 15-inch (2018)
- Processor: 2.6 GHz 6-Core Intel Core i7
- Architecture: Intel x86_64
- Memory: 32 GB 2400 MHz DDR4
- Graphics: Intel UHD Graphics 630
- Operating System: macOS Sequoia 15.7.7

This machine is being used primarily for:

- MuJoCo environment development
- robot model development
- controller implementation
- debugging
- visualization
- Gymnasium environment development
- small-scale RL tests
- experiment analysis
- documentation

Large-scale reinforcement-learning experiments may later be transferred to university computing resources.

---

# 4. Software Environment

The current environment uses:

- Python 3.11
- Python virtual environment (`venv`)
- MuJoCo 3.8.1
- NumPy
- SciPy
- Matplotlib
- Pandas
- Gymnasium
- Git
- GitHub

The Python dependencies are recorded in:

```text
requirements.txt
```

---

# 5. Why MuJoCo 3.8.1 Is Used

The development machine is an Intel-based Mac using the `x86_64` architecture.

An unpinned installation attempt using:

```bash
pip install mujoco
```

attempted to install MuJoCo 3.12.0 from source instead of using a compatible prebuilt Intel macOS wheel.

The source build failed with:

```text
RuntimeError: MUJOCO_PATH environment variable is not set
```

MuJoCo 3.8.1 was therefore installed explicitly:

```bash
pip install "mujoco==3.8.1"
```

This version successfully installed and runs on the current Intel Mac development environment.

The project should therefore keep the MuJoCo version pinned for reproducibility.

---

# 6. Repository

Repository directory:

```text
rnd-dual-arm-marl-benchmark
```

The project is maintained using Git and GitHub.

The repository will contain source code, simulation models, experiment configurations, documentation, and results required to reproduce the benchmark.

---

# 7. Python Virtual Environment Setup

From the repository root, the virtual environment was created using:

```bash
python3.11 -m venv .venv
```

The environment is activated using:

```bash
source .venv/bin/activate
```

When active, the terminal displays:

```text
(.venv)
```

The Python executable can be verified using:

```bash
which python
```

The path should point to:

```text
<repository>/.venv/bin/python
```

---

# 8. Installing MuJoCo

With the virtual environment activated:

```bash
pip install "mujoco==3.8.1"
```

Installation can be verified using:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

Expected version:

```text
3.8.1
```

A second import test is:

```bash
python -c "import mujoco; print('MuJoCo installation successful')"
```

Expected output:

```text
MuJoCo installation successful
```

---

# 9. Installing Additional Python Dependencies

The basic scientific and environment packages were installed using:

```bash
pip install gymnasium scipy matplotlib pandas
```

The complete environment is saved using:

```bash
pip freeze > requirements.txt
```

A future machine can recreate the Python environment using:

```bash
pip install -r requirements.txt
```

The same Python version should preferably be used when reproducing the environment.

---

# 10. MuJoCo Installation Verification

A minimal MuJoCo physics simulation was implemented before introducing a robot model.

The test consisted of:

- a ground plane;
- a free rigid box;
- gravity;
- MuJoCo physics stepping.

The box was initialized above the ground plane.

During simulation, the box:

1. fell under gravity;
2. contacted the ground;
3. remained on the plane.

This verified that:

- MuJoCo could load a model;
- physics stepping worked;
- gravity worked;
- collision/contact handling worked;
- the graphical viewer worked.

Relevant scripts:

```text
scripts/test_mujoco.py
scripts/test_viewer.py
```

On macOS, graphical passive-viewer scripts are executed using:

```bash
mjpython scripts/test_viewer.py
```

---

# 11. Franka Panda Model

The Franka Emika Panda model is based on the MuJoCo Menagerie model collection.

The Menagerie repository was added under:

```text
assets/models/mujoco_menagerie/
```

The Panda model is located at:

```text
assets/models/mujoco_menagerie/franka_emika_panda/
```

Important files include:

```text
panda.xml
scene.xml
assets/
```

The original Menagerie model should remain unmodified.

Project-specific modifications should be made using separate custom model files.

---

# 12. Loading the Panda Robot

The Panda scene was loaded using:

```python
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
```

The model was successfully displayed using the MuJoCo viewer.

Relevant script:

```text
scripts/view_panda.py
```

Run using:

```bash
mjpython scripts/view_panda.py
```

This verified that:

- the Panda model loads correctly;
- mesh assets are available;
- the robot can be rendered;
- the model is compatible with the installed MuJoCo version.

---

# 13. Panda Model Inspection

The Panda model was inspected programmatically.

Relevant script:

```text
scripts/inspect_panda.py
```

The model contains nine joints.

## Arm Joints

```text
joint1
joint2
joint3
joint4
joint5
joint6
joint7
```

These correspond to the seven articulated joints of the Panda arm.

## Gripper Joints

```text
finger_joint1
finger_joint2
```

These correspond to the two gripper fingers.

---

# 14. Panda Actuators

Eight actuators were identified:

```text
actuator1
actuator2
actuator3
actuator4
actuator5
actuator6
actuator7
actuator8
```

The first seven actuators control the seven arm joints.

The eighth actuator controls the gripper mechanism.

Therefore:

```text
7 arm actuators
+
1 gripper actuator
=
8 actuators
```

The number of joints and actuators should not be assumed to be identical.

---

# 15. Panda Body Structure

The following bodies were identified:

```text
world
link0
link1
link2
link3
link4
link5
link6
link7
hand
left_finger
right_finger
```

The body:

```text
hand
```

was initially used as the end-effector reference.

The finger bodies are:

```text
left_finger
right_finger
```

The original Panda model did not contain a site suitable for the project's desired gripper-center task-space reference.

---

# 16. Single-Joint Control Test

A first control experiment commanded one Panda actuator while leaving the remaining arm actuators unchanged.

Relevant script:

```text
scripts/move_one_joint.py
```

The test confirmed that:

- actuator commands can be supplied through `data.ctrl`;
- an individual Panda joint can be controlled;
- the model responds correctly to position commands.

---

# 17. Seven-Joint Control Test

A predefined seven-joint target configuration was subsequently applied.

Relevant script:

```text
scripts/move_all_joints.py
```

An example target configuration used during testing was:

```python
target[0] = 0.0
target[1] = -0.5
target[2] = 0.0
target[3] = -2.0
target[4] = 0.0
target[5] = 1.5
target[6] = 0.8
```

The Panda successfully moved from its initial configuration to the commanded configuration.

This verified that all seven arm joints can be controlled simultaneously.

---

# 18. Robot State Reading

Robot state access was implemented in:

```text
scripts/read_panda_state.py
```

The initial model dimensions were:

```text
nq = 9
nv = 9
nu = 8
```

Where:

- `nq` is the number of generalized position coordinates;
- `nv` is the number of generalized velocity coordinates;
- `nu` is the number of control inputs.

The initial state was:

```text
qpos = [0. 0. 0. 0. 0. 0. 0. 0. 0.]
qvel = [0. 0. 0. 0. 0. 0. 0. 0. 0.]
ctrl = [0. 0. 0. 0. 0. 0. 0. 0.]
```

The individual joint positions were also mapped to their corresponding joint names.

This functionality will later be used when constructing observations for the reinforcement-learning environment.

---

# 19. End-Effector Tracking Using the Hand Body

The Panda `hand` body was initially used as the end-effector reference.

The body ID was obtained using:

```python
hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)
```

Its world position was obtained using:

```python
data.xpos[hand_id]
```

At the initial configuration, the hand position was:

```text
[0.088, 0.000, 0.926]
```

Its orientation was available through:

```python
data.xmat[hand_id]
```

The initial hand rotation matrix was:

```text
[[ 0.70710681  0.70710676  0.        ]
 [ 0.70710676 -0.70710681  0.        ]
 [ 0.          0.         -1.        ]]
```

---

# 20. Tracking the End Effector During Motion

The hand position was tracked while commanding the predefined seven-joint configuration.

Relevant script:

```text
scripts/track_end_effector.py
```

The initial hand position was approximately:

```text
[0.088, 0.000, 0.926]
```

After the robot moved, the position settled at approximately:

```text
[0.387, 0.000, 0.648]
```

Example recorded values:

```text
Step      0 | Hand xyz =  0.088,  0.000,  0.926
Step    500 | Hand xyz =  0.387,  0.000,  0.648
Step   1000 | Hand xyz =  0.387,  0.000,  0.648
Step   1500 | Hand xyz =  0.387,  0.000,  0.648
Step   2000 | Hand xyz =  0.387,  0.000,  0.648
```

This verified the relationship:

```text
joint command
    ↓
joint configuration
    ↓
forward kinematics
    ↓
end-effector Cartesian position
```

This relationship is important for both classical task-space control and reinforcement-learning observations/rewards.

---

# 21. Custom End-Effector Site

A custom project-specific Panda model was created so that the original Menagerie model remains unchanged.

Custom model directory:

```text
assets/models/custom_panda/
```

Custom model:

```text
assets/models/custom_panda/panda_with_site.xml
```

A site named:

```text
ee_site
```

was added to the `hand` body.

The site definition is:

```xml
<site
    name="ee_site"
    pos="0 0 0.10"
    size="0.01"
    rgba="1 0 0 1"
/>
```

The site is attached to the hand and provides a task-space reference around the gripper working region.

---

# 22. End-Effector Site Verification

The site was tested using:

```text
scripts/test_ee_site.py
```

The site ID was obtained using:

```python
site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "ee_site"
)
```

Its world position was obtained using:

```python
data.site_xpos[site_id]
```

The verification output was:

```text
ee_site ID: 0
ee_site position: [0.088 0.    0.826]
```

The site was also successfully visualized as a red marker around the gripper region.

This confirms that the custom task-space reference can be accessed programmatically and visually.

---

# 23. Purpose of the End-Effector Site

The custom end-effector site is intended to support later development of:

- inverse kinematics;
- Cartesian end-effector control;
- grasp-point definition;
- distance-to-object calculations;
- reinforcement-learning observations;
- reward calculations;
- cooperative dual-arm coordination.

In the dual-arm model, separate references will be required, conceptually:

```text
left_ee_site
right_ee_site
```

This will allow the two end-effectors to be tracked independently.

---

# 24. Current Repository Components

At the current development stage, important files include:

```text
rnd-dual-arm-marl-benchmark/
│
├── .venv/
│
├── assets/
│   └── models/
│       ├── mujoco_menagerie/
│       │   └── franka_emika_panda/
│       │
│       └── custom_panda/
│           ├── panda_with_site.xml
│           ├── scene.xml
│           └── assets/
│
├── docs/
│   ├── panda_model_notes.md
│   └── SETUP_AND_PROGRESS.md
│
├── scripts/
│   ├── test_mujoco.py
│   ├── test_viewer.py
│   ├── view_panda.py
│   ├── inspect_panda.py
│   ├── move_one_joint.py
│   ├── move_all_joints.py
│   ├── read_panda_state.py
│   ├── track_end_effector.py
│   └── test_ee_site.py
│
└── requirements.txt
```

The exact repository structure may be expanded as the project progresses.

---

# 25. How to Run the Current Project

## 25.1 Open Terminal

Navigate to the repository:

```bash
cd ~/Documents/RnD/rnd-dual-arm-marl-benchmark
```

## 25.2 Activate the Environment

```bash
source .venv/bin/activate
```

The prompt should begin with:

```text
(.venv)
```

## 25.3 Verify MuJoCo

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

Expected:

```text
3.8.1
```

## 25.4 Run Basic Physics Test

```bash
python scripts/test_mujoco.py
```

## 25.5 Run Viewer Test

```bash
mjpython scripts/test_viewer.py
```

## 25.6 View Panda

```bash
mjpython scripts/view_panda.py
```

## 25.7 Inspect Panda

```bash
python scripts/inspect_panda.py
```

## 25.8 Move One Joint

```bash
mjpython scripts/move_one_joint.py
```

## 25.9 Move All Seven Arm Joints

```bash
mjpython scripts/move_all_joints.py
```

## 25.10 Read Panda State

```bash
python scripts/read_panda_state.py
```

## 25.11 Track End Effector

```bash
mjpython scripts/track_end_effector.py
```

## 25.12 Test Custom End-Effector Site

```bash
mjpython scripts/test_ee_site.py
```

---

# 26. Current Development Status

## Completed

- [x] Repository created
- [x] Python 3.11 environment created
- [x] MuJoCo 3.8.1 installed
- [x] MuJoCo import verified
- [x] Basic physics simulation verified
- [x] MuJoCo viewer verified
- [x] Gravity verified
- [x] Contact/collision with ground verified
- [x] MuJoCo Menagerie obtained
- [x] Franka Panda model loaded
- [x] Panda model visualized
- [x] Panda joints inspected
- [x] Panda actuators inspected
- [x] Panda body structure inspected
- [x] One Panda joint controlled
- [x] Seven Panda arm joints controlled
- [x] Joint positions accessed
- [x] Joint velocities accessed
- [x] Control values accessed
- [x] Hand Cartesian position accessed
- [x] Hand orientation accessed
- [x] End-effector position tracked during motion
- [x] Custom Panda model created
- [x] Custom `ee_site` added
- [x] `ee_site` successfully visualized
- [x] `ee_site` successfully accessed programmatically
- [x] Python dependencies recorded

---

# 27. Current Project Phase

The project is currently transitioning from:

```text
MuJoCo setup and single-arm verification
```

to:

```text
dual-arm cooperative manipulation environment development
```

The installation and single-Panda technical foundation is complete.

The next major milestone is construction of the actual benchmarking environment.

---

# 28. Planned Benchmark Scene

The long-term simulation environment will contain:

- two Franka Panda robotic arms;
- a shared rigid transport object;
- a ground/table workspace;
- a target region;
- independent left and right end-effector references;
- object state information;
- goal state information.

The initial benchmark object will preferably be a simple rigid rectangular box.

A box is selected initially because it provides:

- simple collision geometry;
- clearly defined grasp regions;
- straightforward mass variation;
- straightforward friction variation;
- straightforward orientation/stability measurements;
- easier debugging than a complex object.

A table-like object may later be introduced as an extension if the core benchmark is completed successfully.

---

# 29. Planned Dual-Arm Naming Convention

The two Panda robots must have unique MuJoCo names.

The intended naming convention is:

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

Similarly:

```text
left_actuator1
...
left_actuator8

right_actuator1
...
right_actuator8
```

End-effector references:

```text
left_ee_site
right_ee_site
```

Hand bodies:

```text
left_hand
right_hand
```

Unique naming is necessary to allow independent state access and control for each robotic agent.

---

# 30. Planned Cooperative Transport Task

The intended task is:

1. initialize two Panda arms around a shared rigid object;
2. position the two end effectors at appropriate grasp/support locations;
3. establish cooperative interaction with the object;
4. move the object toward a specified target;
5. maintain object stability during transport;
6. avoid invalid collisions or object dropping;
7. determine task success using measurable criteria.

The same environment and evaluation criteria will ultimately be used for classical control, IPPO, and MAPPO.

---

# 31. Planned Observation Information

The exact RL observation space has not yet been finalized.

Potential state information includes:

```text
left arm joint positions
left arm joint velocities
right arm joint positions
right arm joint velocities
left end-effector position
right end-effector position
object position
object orientation
object velocity
goal position
relative end-effector/object positions
relative object/goal position
```

Only information justified by the final experimental design should be included.

---

# 32. Planned Action Representation

The project proposal specifies joint-level control commands.

The exact RL action representation will be finalized after the dual-arm environment and classical baseline are operational.

Potential options include:

- joint position targets;
- joint position increments;
- joint velocity commands.

The same physical control interface should be used as consistently as possible when comparing learning-based methods.

---

# 33. Planned Evaluation Metrics

The benchmark is expected to include metrics such as:

- task success rate;
- task completion time;
- object stability;
- object position error;
- object orientation/tilt;
- collision behavior;
- learning convergence for RL methods;
- robustness under selected disturbances.

The exact mathematical definitions and acceptance thresholds remain to be finalized.

---

# 34. Planned Robustness Conditions

After nominal transport works, selected robustness experiments may include:

- object mass variation;
- friction variation;
- sensor/state noise;
- action disturbances;
- external perturbations.

The final number of robustness experiments will depend on available development and training time.

The priority is to perform a smaller number of well-controlled and reproducible experiments rather than a large number of incomplete experiments.

---

# 35. Next Development Tasks

The immediate sequence is:

```text
Single Panda foundation complete
        ↓
Create dual-Panda model
        ↓
Assign unique left/right names
        ↓
Place both robots in custom scene
        ↓
Verify both robots load
        ↓
Read both robot states independently
        ↓
Control both robots independently
        ↓
Add transport object
        ↓
Add goal region
        ↓
Define object/grasp reference sites
        ↓
Develop basic task-space/IK control
        ↓
Implement classical cooperative baseline
        ↓
Create Gymnasium/MARL environment
        ↓
Define observations/actions/reward
        ↓
IPPO integration
        ↓
MAPPO integration
        ↓
Benchmarking
        ↓
Robustness experiments
        ↓
Analysis and report
```

---

# 36. Reproducibility Rules

The following rules should be maintained throughout development:

1. Do not directly modify the original MuJoCo Menagerie models.
2. Keep project-specific models in separate directories.
3. Pin important dependency versions.
4. Maintain `requirements.txt`.
5. Avoid hard-coded absolute machine-specific paths.
6. Store experiment parameters in configuration files when the experiment framework is introduced.
7. Record random seeds for RL experiments.
8. Save model checkpoints.
9. Save evaluation results separately from training logs.
10. Keep experiment scripts executable from the command line.
11. Commit meaningful milestones to Git.
12. Document major changes and design decisions.
13. Keep the project transferable to university computing infrastructure.

---

# 37. Current Technical Checkpoint

At this checkpoint, the following pipeline has been demonstrated successfully:

```text
Intel macOS development machine
        ↓
Python 3.11 virtual environment
        ↓
MuJoCo 3.8.1
        ↓
Physics simulation
        ↓
MuJoCo viewer
        ↓
Franka Panda model
        ↓
Joint/actuator inspection
        ↓
Single-joint control
        ↓
Seven-joint control
        ↓
Robot-state access
        ↓
End-effector Cartesian tracking
        ↓
Custom gripper-center site
```

This establishes the technical foundation required to begin development of the dual-arm cooperative transport environment.

---

# 38. Progress Log

## Checkpoint 1 — MuJoCo Environment

Status: **Completed**

MuJoCo was successfully installed and verified using a falling-box simulation.

## Checkpoint 2 — Single Panda Model

Status: **Completed**

The Franka Panda model was successfully imported from MuJoCo Menagerie and visualized.

## Checkpoint 3 — Joint-Level Control

Status: **Completed**

Individual and simultaneous seven-joint position commands were successfully tested.

## Checkpoint 4 — State Access

Status: **Completed**

Joint positions, joint velocities, controls, hand position, and hand orientation can be accessed programmatically.

## Checkpoint 5 — End-Effector Reference

Status: **Completed**

A custom `ee_site` was created, visualized, and accessed programmatically.

## Checkpoint 6 — Dual-Arm Environment

Status: **Next**

The next implementation stage is creation of a two-Panda custom scene for cooperative rigid-object transport.

---

# 39. Important Current Design Decision

The first cooperative transport benchmark will use a simple rigid box rather than a complex object.

The purpose is to establish a stable and reproducible cooperative manipulation task before introducing additional complexity.

A more complex object such as a table may be evaluated later if sufficient project time remains.

---

# 40. Notes for Future Cluster Migration

When moving training to university computing infrastructure:

1. clone the GitHub repository;
2. use a compatible Python version;
3. create a fresh virtual/Conda environment;
4. install dependencies from the project's dependency specification;
5. verify the MuJoCo version;
6. run non-rendered simulation tests;
7. verify the RL framework separately;
8. avoid requiring the interactive viewer for training;
9. run a short smoke test before submitting long jobs;
10. record cluster-specific setup separately from the core project environment.

Training code should eventually support headless execution so that the simulation does not require an interactive graphical viewer on the cluster.

---

# 41. Documentation Maintenance

This document should be updated whenever a major milestone is completed.

In particular, future sections should document:

- dual-Panda model construction;
- custom benchmark scene;
- object model;
- grasp configuration;
- inverse-kinematics implementation;
- classical controller;
- Gymnasium/MARL interface;
- observation definition;
- action definition;
- reward function;
- termination conditions;
- IPPO implementation;
- MAPPO implementation;
- training configuration;
- evaluation protocol;
- robustness experiments;
- results;
- known limitations;
- cluster configuration.

The purpose is for another researcher, supervisor, or future developer to understand not only how to run the final code, but also how the experimental system was constructed and why the major design decisions were made.
