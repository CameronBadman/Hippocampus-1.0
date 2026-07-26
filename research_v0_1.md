# Spider v0.1 AutoResearch protocol

## Goal

Determine whether exact train/runtime controller alignment enables the fixed
tied recurrent Spider to outperform a pooled control in autonomous graph
execution.

Machine records are append-only JSONL under
`artifacts/spider_v0_1/experiments.jsonl`; the generated human summary is
`artifacts/spider_v0_1/EXPERIMENT_SUMMARY.md`.

## Success metric

Primary score is autonomous success on complete non-sealed validation splits:
answerable cases require correct `ANSWER`, a valid trace, and required
evidence; unanswerable cases require the correct unknown family and no false
answer. Higher is better.

Success requires:

- improved mean autonomous success versus E000;
- three-seed recurrent mean above the matched pooled mean;
- zero deterministic replay and row-permutation decision mismatches.

A negative comparison is a valid outcome and must remain in the ledger.

## Constraints

- Evaluator:
  `.venv/bin/python scripts/spider_v0_1_evaluator.py --config <config>`
- Keep policy: `pass_only` for the fixed matrix; no registered result is
  reverted or hidden.
- `pause_every: never`
- `max_iterations: 10` accepted records, including E000.
- Five-minute timeout per experiment, enforced with `timeout 5m`.
- `noise_runs: 1` during the fixed sequence; three independent finalist seeds.
- `min_delta: 0`
- Guard: full applicable test suite, finite metrics, zero replay mismatch,
  zero row-permutation decision mismatch, and zero sealed accesses during
  search.
- Forbidden changes: packed manifold substrate, renderer semantics, model
  widths, compositional attention, swaps, language writers, reinforcement
  learning, and all files under `docs/spider_v0/` or
  `artifacts/spider_v0/`.

## Current approach

The v0 tied recurrent standard-attention Spider scored candidates well under
teacher forcing but stopped after one autonomous round on average and had
sealed evidence F1 0.0232. The failure audit confirms ten controller/training
mismatches. v0.1 replaces three rollout implementations with one proposal,
action, transition, and termination state machine.

## Search space

Only the frozen causal sequence is permitted:

1. E000 old checkpoint under the corrected controller;
2. E001 unified transition with oracle actions;
3. E002 scheduled independent closed-loop actions;
4. E003 balanced and set-level evidence loss;
5. E004 hierarchical termination;
6. E005 matched pooled control;
7. two additional seeds for E004 and E005.

## Evaluator protocol

Each accepted training run uses the complete 512-case v0.2 train split and all
non-sealed validation cases. The evidence threshold is calibrated on
validation ID only. Every record includes source commit/hash, dataset and
split digest, exact config, parameter count, steps, seed, autonomous and
diagnostic metrics, runtime, peak memory, checkpoint hash, action-source
counts, threshold, and failure reason.

The evaluator emits:

```json
{"pass": true, "status": "accepted", "score": 0.0, "metrics": {}}
```

## Environment tier

Tier 1: Bash, Python, PyTorch CUDA, deterministic tests, and mechanical
evaluation are available. The loop runs unattended and does not pause.

## History

No v0.1 experiment has run. E000 begins only after the unified controller,
v0.2 manifests, metrics, and guard tests are committed.
