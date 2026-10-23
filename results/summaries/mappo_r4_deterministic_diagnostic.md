# MAPPO R4 Deterministic Policy Diagnostic

## Experimental configuration

- Algorithm: MAPPO
- Training seed: 42
- Training steps: 10,000
- Action mode: Cartesian, 3D per arm
- Target position: [0.05, 0.0, 0.58] m
- Success tolerance: 0.01 m
- Progress reward weight: 100
- Evaluation mode: deterministic (tanh of Gaussian mean)
- Checkpoint: checkpoints/mappo/mappo_cartesian_r4_10k_5cm.pt

## Observed results

- Episode length: 500 steps
- Success: False
- Closest target distance: 14.23 mm at step 360
- Final target distance: 22.78 mm
- Final forward displacement: 61.42 mm
- Final linear speed: 0.00570 m/s
- Final orientation error: 0.25123 rad

## Action behavior

At step 360:
- Left X action: +0.23207
- Right X action: +0.25584
- Forward displacement: 45.01 mm
- Orientation error: 0.18573 rad

At step 500:
- Left X action: +0.23103
- Right X action: +0.23589
- Forward displacement: 61.42 mm
- Orientation error: 0.25123 rad

Both actors continued requesting positive X motion after the
closest approach. The object continued moving forward and its
distance from the target increased.

## Interpretation

The deterministic policy approaches the transport target but does
not adequately stabilize near it. Its closest distance remains
outside the 10 mm success tolerance. Orientation error also
exceeds the 0.20 rad threshold later in the episode.

The evidence supports a persistent forward-command behavior.
The underlying training-related cause has not yet been established.

## Limitations

- Single trained checkpoint
- One nominal deterministic trajectory, reproduced across resets
- No randomized initial-state evaluation
- Stochastic success does not imply deterministic success

## Next experiments

1. Compare deterministic and stochastic trajectory behavior.
2. Investigate policy action distributions and variance.
3. Evaluate robustness across randomized initial conditions.
4. Benchmark classical control, IPPO and MAPPO under matched tasks.
