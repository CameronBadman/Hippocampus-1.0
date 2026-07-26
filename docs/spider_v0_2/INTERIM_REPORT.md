# Spider v0.2 Interim Result

## Status

This is a post-sealed architectural diagnostic, not a new sealed evaluation.
The historical Spider v0 and v0.1 results remain unchanged.

One complete matched seed pair is certified. A second recurrent run reached a
verified 3,000-step checkpoint before being stopped for the user-requested
handoff. It has no evaluation result and is excluded from comparisons.

| Run | Status | Fixed-horizon structural success | Exact evidence set |
| --- | --- | ---: | ---: |
| Recurrent, seed 1701 | accepted, 6,000 steps | 0.3828125 | 0.3828125 |
| Pooled, seed 1701 | accepted, 6,000 steps | 0.4609375 | 0.4609375 |
| Recurrent, seed 1802 | excluded, stopped at 3,000 steps | not evaluated | not evaluated |

The paired seed-1701 recurrent-minus-pooled delta is `-0.078125`. Pooled also
has higher evidence average precision (`0.6823` versus `0.4772`) and evidence
F1 (`0.6667` versus `0.6190`). This single pair is evidence against a recurrent
advantage, but it is not the preregistered three-seed conclusion.

## Is recurrent state being used?

Yes, for the accepted recurrent seed:

| Intervention | Structural success | Delta from intact |
| --- | ---: | ---: |
| Intact | 0.3828125 | 0.0000000 |
| Detach between rounds | 0.3828125 | 0.0000000 |
| Reset each round | 0.0000000 | -0.3828125 |
| Shuffle across hypotheses | 0.0859375 | -0.2968750 |
| Replace with pooled current node | 0.0625000 | -0.3203125 |

Detach is a gradient intervention and therefore should not alter a pure
forward evaluation. Resetting, shuffling, and replacing state do alter the
forward computation, and their large degradation shows that this checkpoint
uses its recurrent path state. It does **not** show that recurrence is better
than pooling: the matched pooled model still wins this seed.

## Integrity

- Frozen model source: `acb533666d481daf9b6fb56562d69a5dd78c5e0e`
- Training manifest:
  `ff36529a8090581f6156a8fc36258e4a14eee9a542955623b70550001469fe56`
- Validation manifest:
  `67c2273e4899af179bc1e10185742b806d751f5f5dba858c771f2eca8a6af4aa`
- Accepted runs report zero sealed access, zero deterministic replay
  mismatches, zero row-permutation decision mismatches, valid paths, and valid
  traces.
- The local CPU suite passed: 170 tests, with seven CUDA-dependent tests
  skipped because CUDA is not visible locally.
- Every accepted and partial periodic checkpoint listed in
  `artifacts/spider_v0_2/GOOGLE_DRIVE_BACKUP.json` was size-, parent-, and
  metadata-verified on Drive.

The verified backup folder is
[Spider-v0.2-Fixed-Horizon](https://drive.google.com/drive/folders/1A8QnvZKDSWeiTXvi6RwYx76LFVAcDZRw).

## Current scientific position

The manifold substrate continues to support deterministic, exchangeable, and
differentiable graph execution. The fixed-horizon diagnostic establishes, for
one seed, that recurrent state carries long-horizon information when premature
termination is suppressed. It has not established that the recurrent
processor earns its additional complexity: the matched pooled control is
better on the first complete seed pair despite having fewer parameters
(`200,549` versus `297,049`).

Do not claim a final architecture result until seeds 1802 and 1903 are rerun
and the fail-closed six-run aggregator accepts the complete matrix.
