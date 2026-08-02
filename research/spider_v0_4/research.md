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

- Phase B evaluator: `.venv/bin/python scripts/run_spider_v0_4_autoresearch.py`
- Phase D evaluator: `.venv/bin/python scripts/run_spider_v0_4_readout.py`
- Keep policy: `pass_only`
- `pause_every: never`
- `max_iterations: 51`
- `noise_runs: 3`
- `min_delta: 0.02`
- Development accelerator: one local NVIDIA GeForce RTX 5070 Ti, FP32.
- Final replication: one A100 only after one pooled and one Spider finalist are
  frozen.
- Phase B dataset: immutable `spider-programs-v0.4-aligned-dev`.
- Post-audit dataset: `spider-programs-v0.4.1-aligned-evidence-dev`, which
  removes the unsupported-query cardinality leak without altering Phase B.
- Renderer: `renderer-v0.4` with the passing A2 orthogonal geometry.
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
| 2 | Cross-modal identity alignment alone improves pooled-model lookup, reachability, and exact evidence recovery under E0. | rejected | 0.5924 | B2 won 0/3 matched seeds. Mean lookup/reachability recall did not clear the +0.30/+0.20 gates; precision and scored coverage also missed the primary constraints in aggregate. Route to generator/label and frozen-logit set-decoding audit; do not run Phase C yet. |
| 3 | The Phase B failure is caused by inconsistent evidence labels. | rejected | 0.0 | All 10,240 cases verify, exact candidate labels match required evidence, positive summary identity is complete, and an observable lookup rule is 1.000 accurate. A separate query-cardinality leak in unsupported cases requires a dataset amendment but does not explain answerable lookup recall. |
| 4 | Frozen B2 ranking has enough information that cardinality-aware decoding can materially improve exact recovery. | kept | 0.8822 | Oracle-cardinality top-k improves mean B2 exact set by 0.2897; every seed clears the +0.15 branch gate. Ranking remains imperfect, especially on positive lookup, so readout and decoding both remain live suspects. |
| 5 | Remove the unsupported-query row-count leak before evidence readout experiments. | kept | 0.5 | `spider-programs-v0.4.1-aligned-evidence-dev` has chance-level query-cardinality answerability accuracy in every partition, zero invalid traces, new seeds, and zero sealed access. |
| 6 | A dedicated or slot-aware evidence readout recovers signal hidden by the shared mean-pooled head. | rejected | 0.7516 | D1, D3, and D4 each won 0/3 matched-seed gates. D4 versus D3 also won 0/3. The readouts improve some precision/AP values but do not materially improve exact recovery or recall; D0 and D2 controls advance. |
| 7 | Split slow D4 evaluation from completed training without altering scientific state. | kept | 0.0 | D4/1701 exceeded the 300-second monolithic process bound after training and all four checkpoint selections. Its timeout is preserved. A tested pause/resume boundary now runs calibration and development evaluation separately while binding the checkpoint and training-source commit. |
| 8 | Resume an interrupted eight-checkpoint finalist selector without accepting a partial comparison. | kept | 0.0 | D2/1701 reached the guard after six selections. A CUDA-tested selector resume reuses completed step records, evaluates only missing checkpoints, and refuses to begin calibration until it writes a complete selected-checkpoint pause. |
