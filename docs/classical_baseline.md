# Classical Cooperative Transport Baseline

## Status

The nominal classical cooperative dual-arm object-transport task has been validated end-to-end in MuJoCo.

Final validated sequence:

1. Approach
2. Bilateral grasp
3. Cooperative lift
4. Flat suspended transport
5. Flat placement
6. Controlled gripper release
7. Outward bilateral arm retreat

The reproducible baseline entry point is:

```bash
python scripts/run_classical_baseline.py
```

A successful run ends with:

```text
CLASSICAL BASELINE: PASS
```

## Model

The task uses two Franka Panda manipulators and the generated transport scene:

```text
assets/models/dual_panda/transport_scene.xml
```

The controller applies MuJoCo bias-force feed-forward compensation to both robot arms during every physics step.

## Validated Task Geometry

Suspended transport target:

```text
X = 0.380 m
Z = 0.580 m
```

Flat placement end-effector reference:

```text
X = 0.408 m
Z = 0.566 m
```

Benchmark goal center:

```text
X = 0.420 m
Y = 0.000 m
```

The placement target is intentionally offset from the goal center. During controlled gripper release the object moves slightly backward before settling, while still remaining fully inside the goal region.

## Goal Containment

Goal half-size:

```text
X = 0.220 m
Y = 0.150 m
```

Object half-size:

```text
X = 0.200 m
Y = 0.120 m
```

For an approximately flat and aligned object, full-footprint containment therefore requires:

```text
|object_x - goal_x| <= 0.020 m
|object_y - goal_y| <= 0.030 m
```

## Trajectories

Transport:

```text
43 Cartesian IK waypoints
100 physics steps per waypoint segment
```

Placement:

```text
15 Cartesian IK waypoints
100 physics steps per waypoint segment
```

Retreat:

```text
21 Cartesian IK waypoints
100 physics steps per waypoint segment
```

Transport and placement use linear joint interpolation between consecutive IK waypoints.

The retreat uses smoothstep interpolation between consecutive IK waypoints.

## Release

The grippers are gradually opened while both arms remain fixed at the placement configuration.

```text
release steps:           3000
post-release hold steps: 2000
```

The release trajectory uses cubic smoothstep interpolation of the gripper command.

## Final Acceptance Criteria

A successful complete task requires:

- object inside the goal region in X
- object inside the goal region in Y
- table vertical support >= 18 N
- residual finger normal force <= 0.5 N
- final object linear speed <= 0.01 m/s
- final object angular speed <= 0.10 rad/s
- absolute final roll and pitch <= 2 degrees
- object displacement during retreat <= 0.005 m

The `0.5 N` finger-force criterion applies to the final residual contact state. Transient finger contact during the retreat is recorded as a diagnostic but is not currently a final-task acceptance criterion.

## Reference Validation Run

A validated nominal baseline run produced:

```text
CLASSICAL BASELINE: PASS

final object position:
[0.403565, -0.000045, 0.559892] m

final object RPY:
[~0.0, ~0.0, -0.159073] deg

goal-center XY error:
0.0164353 m

final table support:
19.62 N

final finger vertical support:
0.0 N

final finger normal force:
0.0 N

final finger contacts:
0

release object X shift:
approximately -4.526 mm

first step with table support >= 18 N:
3076

first step with zero finger contacts:
3110

final retreat displacement:
approximately 0.106 mm

maximum retreat displacement:
approximately 0.106 mm

maximum transient retreat finger normal force:
approximately 0.553 N

final linear speed:
approximately 1.64e-14 m/s

final angular speed:
approximately 1.04e-15 rad/s

simulated task time:
59.2 s

wall-clock execution time on the validation machine:
approximately 25.95 s
```

The final object is stable and fully supported by the table after both arms have retreated.

## IK Validation

The final baseline generated and validated all required pose-IK trajectories before starting the dynamics run.

Representative validation results:

```text
Pregrasp IK: PASS
Grasp IK: PASS
Lift IK: PASS

Horizontal transport trajectory:
43 waypoints
max IK position error:    ~0.0996 mm
max IK orientation error: ~0.0050 deg

Flat-placement trajectory:
15 waypoints
max IK position error:    ~0.0996 mm
max IK orientation error: ~0.0048 deg

Outward retreat trajectory:
21 waypoints
max IK position error:    ~0.1000 mm
max IK orientation error: ~0.0052 deg
```

## Phase Validation

The validated end-to-end run produced:

```text
APPROACH    PASS
GRASP       PASS
LIFT        PASS
TRANSPORT   PASS
PLACE       PASS
RELEASE     PASS
RETREAT     PASS
```

Important intermediate results included:

```text
Grasp:
finger contacts:               64
finger normal-force sum:       ~99.60 N

Lift:
object Z change:               ~0.01414 m
table support:                 0.0 N
finger vertical support:       ~19.62 N

Transport:
final object x:                ~0.380022 m
table contact occurred:        False
final orientation change:      ~0.09784 deg

Placement:
object x:                      ~0.408030 m
table vertical support:        ~1.98 N
finger vertical support:       ~17.64 N
inside goal X:                 True
inside goal Y:                 True

Release:
final table support:           19.62 N
final finger vertical support: 0.0 N
final finger normal force:     0.0 N
final finger contacts:         0

Retreat:
final object displacement:     ~0.000106 m
maximum object displacement:   ~0.000106 m
```

## Files

Reusable low-level controller:

```text
src/controllers/classical_controller.py
```

Final benchmark runner:

```text
scripts/run_classical_baseline.py
```

Detailed historical validation / diagnostic experiment:

```text
scripts/test_cooperative_place_release.py
```

Core controller smoke test:

```text
scripts/test_classical_controller_core.py
```

## Benchmark Metrics

The current runner additionally records descriptive quantities such as:

- goal-center XY error
- maximum object roll/pitch
- maximum object linear speed
- maximum object angular speed
- actuator-force-squared integral
- joint-command variation integral

The validated nominal run produced:

```text
max |roll/pitch|:
~0.10307 deg

max object linear speed:
~0.06792 m/s

max object angular speed:
~0.10089 rad/s

actuator-force-squared integral:
~3123.17

joint-command variation integral:
~0.001523
```

These are currently descriptive raw metrics.

Their exact definitions for the final comparison with IPPO and MAPPO will be fixed in the common benchmark/evaluation layer so that all methods are measured identically.

## Milestone Status

The nominal classical-controller milestone is complete.

The next development stage is the common dual-arm MARL environment and evaluation interface so that the classical controller, IPPO, and MAPPO can be evaluated under the same task dynamics, observations, success criteria, and benchmark metrics.
