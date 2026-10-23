# R3 vs R4 Reward Sensitivity Experiment

## Setup
- Environment: MuJoCo cooperative dual-arm object transport
- Algorithms: IPPO and MAPPO
- Action representation: Cartesian (3D per arm)
- Target displacement: 0.05 m
- Success tolerance: 0.01 m
- Training budget: 10,000 environment steps per run
- Training seed: 42
- Evaluation seed: 100
- Stochastic evaluation: 20 episodes per checkpoint
- Deterministic evaluation: 5 episodes per checkpoint
- R3 progress reward weight: 50
- R4 progress reward weight: 100
- Other reward coefficients unchanged

## Results

| Model | Training successes | Stochastic successes | Deterministic successes |
|---|---:|---:|---:|
| IPPO R3 | 6/22 | 0/20 | 0/5 |
| IPPO R4 | 13/24 | 0/20 | 0/5 |
| MAPPO R3 | 0/20 | 0/20 | 0/5 |
| MAPPO R4 | 0/20 | 16/20 | 0/5 |

## MAPPO R4 Evaluation
- Stochastic success rate: 80%
- Mean stochastic final distance: 10.86 mm
- Mean stochastic closest distance: 10.28 mm
- Mean stochastic forward displacement: 45.24 mm
- Stochastic cooperative contact maintained throughout all 20 episodes
- Deterministic success rate: 0%
- Mean deterministic final distance: 22.78 mm
- Mean deterministic closest distance: 14.23 mm
- Mean deterministic forward displacement: 61.42 mm
- Mean deterministic final orientation error: 0.25123 rad
- Deterministic contact-loss steps: 0

## Interpretation
Increasing the progress reward from 50 to 100 improved the
evaluated stochastic performance of MAPPO in this experiment,
but did not improve the evaluated performance of IPPO.

MAPPO R4 retained cooperative grasp but did not succeed
under deterministic action selection.

Training success rates did not reliably predict final-policy
evaluation performance.

## Limitations
- One training seed per configuration
- Fixed or nearly fixed initial environment state
- Small evaluation sample
- No randomized robustness evaluation yet
- Stochastic and deterministic evaluation results differ
- These results do not establish general algorithm superiority

## Next Steps
1. Reproducibility across multiple training seeds
2. Initial-state and target-position robustness
3. Classical vs IPPO vs MAPPO benchmark
4. Final plots, statistical summaries, and report
