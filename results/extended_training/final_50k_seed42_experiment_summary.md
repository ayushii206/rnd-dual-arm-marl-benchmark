# Final MARL Training-Budget Experiment — Seed 42

## Research context

Project: Benchmarking Classical Control vs Decentralized Multi-Agent RL for Cooperative Dual-Arm Object Transport.

Experiment branch: experiment/final-marl-training-improvement

Objective: Test whether increasing the training budget from 10,000 to 50,000 environment steps improves the final deterministic IPPO and MAPPO policies.

## Experimental configuration

- Algorithms: IPPO and MAPPO
- Training initialization: Fresh networks, trained independently
- Training seed: 42
- Training steps: 50,000
- Rollout length: 512
- Action representation: Cartesian, 3D per agent
- Target displacement: 0.05 m
- Success distance: 0.01 m
- Training progress reward weight: 100
- PPO architecture, reward structure and environment: Unchanged
- Evaluation: Deterministic tanh of actor mean
- Evaluation seed: 100
- Evaluation episodes per checkpoint: 1
- Maximum episode duration: 500 steps

## Training results

| Metric | IPPO | MAPPO |
|---|---:|---:|
| Environment steps | 50,000 | 50,000 |
| Completed training episodes | 107 | 130 |
| Successful training episodes | 21 | 62 |
| Stochastic training episode success rate | 19.6% | 47.7% |
| PPO rollout updates | 98 | 98 |

Both training histories showed successful episodes followed by periods of deteriorating performance. Training success rates summarize changing policies and must not be interpreted as final-policy evaluation success rates.

## Final deterministic evaluation

| Metric | IPPO 50k | MAPPO 50k |
|---|---:|---:|
| Success | False | False |
| Episode steps | 500 | 500 |
| Final distance (m) | 0.02694985 | 0.03252769 |
| Minimum distance (m) | 0.01666627 | 0.00728103 |
| Step at minimum distance | 334 | 199 |
| Both grasps maintained at minimum distance | False | False |
| Final orientation error (rad) | 0.43165499 | 0.30618892 |
| Final linear speed (m/s) | 0.00345389 | Approximately zero |
| Final angular speed (rad/s) | 0.01041095 | Approximately zero |
| Left missing-contact steps | 0 | 312 |
| Right missing-contact steps | 200 | 191 |
| Both missing-contact steps | 0 | 191 |
| Dropped | False | False |

The IPPO final policy never entered the 10 mm positional success region. It also lost right-arm grasp contact and exceeded the final orientation tolerance.

The MAPPO final policy temporarily entered the positional target region, but bilateral grasp contact was absent at its closest approach. Both grasp contacts were absent at the final step, and final position and orientation exceeded their tolerances.

## Verified baseline comparison

The authoritative original five-seed stability benchmark is:
results/comparison/multiseed_deterministic_5cm_stability.csv

| Metric | IPPO 10k | IPPO 50k | MAPPO 10k | MAPPO 50k |
|---|---:|---:|---:|---:|
| Training seed | 42 | 42 | 42 | 42 |
| Deterministic success | False | False | False | False |
| Final distance (mm) | 30.7694 | 26.9499 | 22.7792 | 32.5277 |
| Episode steps | 500 | 500 | 500 | 500 |

IPPO final target error decreased by approximately 12.4%, but the policy still failed.

MAPPO final target error increased by approximately 42.8%, and bilateral grasp maintenance deteriorated: the original 10k checkpoint maintained both grasps throughout, while the 50k checkpoint experienced substantial contact loss.

Neither 50k policy improved deterministic task success for seed 42.

## Scientific limitations

1. Only one independent training seed was tested at 50k for each algorithm.
2. Each final policy received one deterministic evaluation episode from a fixed initialized state.
3. Training histories were printed to terminal; intermediate best-performing policies were not checkpointed.
4. Stochastic training success is not interchangeable with deterministic final-policy performance.
5. The evaluation environment uses its default progress reward weight of 50, not the training value of 100. This changes reported evaluation returns.
6. Training-budget effects cannot be generalized across seeds from these results alone.

## Artifacts

IPPO checkpoint:
checkpoints/ippo/ippo_r4_seed42_fresh50k.pt

MAPPO checkpoint:
checkpoints/mappo/mappo_r4_seed42_fresh50k.pt

IPPO evaluation CSV:
results/extended_training/ippo_seed42_fresh50k_deterministic_eval.csv

MAPPO evaluation CSV:
results/extended_training/mappo_seed42_fresh50k_deterministic_eval.csv

Note: Checkpoints are Git-ignored and require separate preservation if transferred to another machine.

## Recommended next actions

- Verify original MAPPO seed-42 10k baseline metrics.
- Preserve the experiment summary and two evaluation CSVs.
- Do not overwrite or discard either 50k checkpoint.
- Return to the main R&D development chat for validation, integration decisions, scientific reporting and advisor-meeting preparation.
- Do not merge, commit or push without explicit review.
