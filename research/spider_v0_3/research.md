# Spider v0.3 evidence and controller research

## Goal

Determine whether exact evidence-pipeline diagnostics, within-case evidence
ranking, disjoint calibration, and directly labelled factorized termination
can make the pooled recurrent Spider execute autonomously without sacrificing
precision or fixed-horizon structural behaviour.

If autonomous pooled execution passes the registered gates, test whether
multi-set attention is useful on a deliberately mean/max-ambiguous
multi-binding benchmark.

## Success metric

Evidence phase:

- scored-positive coverage at least 0.95;
- conditional selection recall at least 0.90 as a diagnostic target;
- an experimental evidence arm advances only under the registered
  recall/exact-set, precision, coverage, and seed gates.

Termination phase:

- useful-state continuation recall at least 0.95;
- premature stops below 0.25;
- autonomous success at least 85% of fixed-horizon success;
- answered-risk degradation no greater than 0.02;
- macro unknown-reason recall at least 0.70.

Architecture phase:

- multi-set attention must beat the strongest pooled control by at least 0.02
  in paired mean and on at least two of three seeds.

## Constraints

- Evaluator: `.venv/bin/python scripts/spider_v0_3_evaluator.py`
- Keep policy: `pass_only`
- `pause_every: never`
- `max_iterations: 39` accepted training runs
- `noise_runs: 3`
- `min_delta: 0.02`
- One sequential A100 session; FP32; seeds 1701, 1802, and 1903.
- All failed, invalid, interrupted, and negative experiments remain logged.
- Spider v0.2 files listed in
  `artifacts/spider_v0_3/V0_2_FREEZE.json` are immutable.
- No sealed Spider split may be loaded, materialised, calibrated on, or
  evaluated.
- No natural-language writers, reinforcement learning, relation IDs, custom
  CUDA kernels, or substrate redesign.

## Current approach

The preserved v0.2 pooled runs achieved evidence recall between 0.64 and 0.78
and exact-set accuracy near 0.45 under a fixed horizon, but existing
"conditioned" metrics are node-only and cannot distinguish three converging
edge actions. The new work first fixes measurement, then changes one evidence
objective at a time, then trains termination heads on direct state labels.

## Search space

1. E0: current balanced BCE plus set objective.
2. E1: class-weighted BCE plus top-scoring hard-negative ranking.
3. E2: E1 plus structurally plausible hard-negative mining.
4. T0: hierarchical six-way control.
5. T1: direct factorized control.
6. T2: factorized control plus per-hypothesis NULL.
7. T3: conditional joint fine-tuning.
8. Conditional multi-binding comparison across mean/max, DeepSets,
   multi-set attention, and exchangeable slot memory.

## Immutable history

The exact historical hashes and tags are recorded in
`artifacts/spider_v0_3/V0_2_FREEZE.json`. New results are post-v0.2
development evidence and cannot alter the certified v0.2 conclusion.

## History

| Iteration | Hypothesis | Status | Score | Notes |
|---:|---|---|---:|---|
| 0 | Freeze history and instrument before changing objectives. | pass | 1.0 | v0.2 tagged; exact edge-aware funnel and independent evidence action tested. |
| 1 | Candidate coverage, rather than scorer selection, is the preserved-run bottleneck. | fail | 0.7031 | Reachable/scored coverage was 1.0; pooled conditional selection recall averaged 0.7031, so the coverage hypothesis was falsified. |
| 2 | Weighted BCE plus multi-positive hard-negative ranking improves evidence recall without losing precision. | fail | 0.5139 | E1 lost 0.1111 recall and 0.0521 exact-set accuracy versus E0 across three matched 1k screens. |
| 3 | Additional structurally plausible negatives improve over ordinary hard negatives. | invalid | 0.5139 | E2 received no additional plausible-negative targets; its model tensors were byte-identical to E1, so it is non-informative. |
| 4 | Longer E0 training improves the calibrated evidence operating point. | fail | 0.5764 | Three 6k continuations averaged lower recall and exact-set accuracy than the 1k E0 screens; calibration exposed overconfident logits. |
| 5 | Direct factor labels restore autonomous continuation while retaining fixed-horizon success. | fail | 0.3490 | T1 matched T0 autonomous success, with 0.6574 continuation recall and 0.5726 retention. |
| 6 | Per-hypothesis NULL reduces false answers without harming structural success. | fail | 0.3021 | T2 lowered false answers to 0.1042 but also reduced fixed-horizon and autonomous success. |

The termination gate failed, so T3 joint fine-tuning and the conditional
multi-binding architecture comparison were not opened. All scores above are
non-sealed development results from the local RTX 5070 Ti contingency; the
registered A100 execution remains unavailable because Colab rejected the
accelerator allocation.
