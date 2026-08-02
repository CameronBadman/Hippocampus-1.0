# Spider v0.4 representation and exact-evidence research

## Goal

Determine whether cross-modal symbol identifiability, training-case diversity,
evidence-specific readout, or set decoding is the limiting cause of Spider's
exact evidence failures. Model width, path capacity, processor depth,
termination, compositional edges, and language writers remain fixed.

## Success metric

Primary: development exact evidence-set accuracy, subject to evidence precision
at least 0.90 and scored-positive coverage at least 0.98.

Tie-breakers are evidence recall, macro average precision, fewer false
positives per case, better worst-positive rank, and earlier checkpoint, in that
order. All metrics are stratified by lookup, reachability, latest-valid, and
corroboration.

## Constraints

- Evaluator: `.venv/bin/python scripts/run_spider_v0_4_autoresearch.py`
- Keep policy: `pass_only`
- `pause_every: never`
- `max_iterations: 51`
- `noise_runs: 3`
- `min_delta: 0.02`
- Development accelerator: one local NVIDIA GeForce RTX 5070 Ti, FP32.
- Final replication: one A100 only after one pooled and one Spider finalist are
  frozen.
- Dataset: `spider-programs-v0.4-aligned-dev`; renderer: `renderer-v0.4`.
- Seeds: 1701, 1802, and 1903.
- Maximum trainable steps: 2,000; checkpoint interval: 250.
- No existing sealed split may be opened, materialised, or used for selection.
- Historical files named in `artifacts/spider_v0_4/V0_3_FREEZE.json` are
  immutable.
- Guards: finite metrics, precision/coverage constraints where applicable,
  zero deterministic replay mismatches, zero row-permutation decision
  mismatches, nonzero hard-negative targets for G2/G3, and all tests passing.
- Forbidden changes: larger `d_model`, additional path rows or Spider blocks,
  compositional edge transforms, learned termination, multi-binding tests,
  natural-language writers, reinforcement learning, and sealed evaluation.

## Current approach

The historical renderer samples symbol vectors from `(modality, symbol)`, so
identity has no learnable cross-modal geometry. The shared candidate head also
mean-pools every path row before all seven outputs. Spider v0.4 tests those
interfaces before changing capacity.

B0 reuses the three immutable 1,000-step v0.3 E0 checkpoints and reevaluates
them under v0.4 instrumentation. This preserves the six-new-training-run
renderer budget for B1 and B2 while providing matched historical A0 controls.

## Search space

1. A0 independent, A1 shared-additive, and A2 fixed-orthogonal renderers.
2. Fixed 512, fixed 8,192, and online/effectively-unbounded training cases.
3. Shared, dedicated pooled, and slot-aware evidence readouts.
4. Frozen-logit P0 global threshold, P1 oracle cardinality, P2 per-case
   threshold, and P3 oracle null diagnostics.
5. Exactly one downstream branch: F0-F3 set decoding when the oracle-k ceiling
   is strong, otherwise G0-G3 clean ranking with guaranteed negatives.

## History

| Iteration | Hypothesis | Status | Score | Notes |
|---:|---|---|---:|---|
| 0 | Freeze protocol before implementing the aligned renderer. | kept | 0.0 | v0-v0.3 artifacts remain immutable; no sealed access is permitted. |
| 1 | A fixed orthogonal cross-modal geometry makes opaque identity recoverable without collapsing modalities. | kept | 1.0 | A2 achieved 1.000 AUROC and 1.000 Top-1 at 64 and 256 unseen-symbol distractors; A0 remained near chance. Graph training is unlocked. |
