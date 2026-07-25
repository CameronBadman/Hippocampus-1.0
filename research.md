# Spider v0 AutoResearch protocol

## Goal

Determine whether a tied recurrent multi-set Spider improves structural graph
program execution over pooled and flat position-free baselines, especially on
path-length, graph-size, topology, cardinality, and equivalent-view shifts.

This file is both the pre-registration and the append-only human-readable
research log. Machine-readable records live in
`artifacts/spider_v0/experiments.jsonl`.

## Frozen inputs

- Generator version: `spider-programs-v0.1`
- Split manifests: `artifacts/spider_v0/splits/*.json`
- Split digests: recorded before the first accepted experiment
- Metric implementation: frozen before the first accepted experiment
- Architecture source baseline: commit containing the first green tiny-overfit
  run
- Sealed-test policy: never load `test_sealed` during search

## Mechanical evaluator

The evaluator command is:

```bash
.venv/bin/python scripts/spider_autoresearch_evaluator.py \
  --config configs/spider_v0/recurrent_multiset.json
```

It must emit one JSON object containing at least:

```json
{
  "status": "accepted",
  "score": 0.0,
  "metrics": {},
  "runtime_seconds": 0.0,
  "peak_memory_bytes": 0
}
```

An experiment is valid only if:

1. the process exits successfully within five minutes;
2. CPU guard tests pass;
3. deterministic replay mismatches are zero;
4. row-permutation decision mismatches are zero;
5. no non-finite loss or metric is recorded;
6. the sealed split was not opened.

Invalid and crashed experiments do not consume the accepted-experiment budget
but remain in the ledger with a failure reason.

## Primary score

The architecture-search score is computed only from validation data:

```text
0.20 * ID decision accuracy
+ 0.20 * graph-size OOD decision accuracy
+ 0.20 * path-length OOD decision accuracy
+ 0.10 * topology OOD decision accuracy
+ 0.10 * cardinality OOD decision accuracy
+ 0.10 * equivalent-view OOD decision accuracy
+ 0.10 * composition OOD decision accuracy
- 0.05 * invalid expansion rate
- 0.02 * normalised context-read cost
```

All components are recorded separately. The score never includes sealed-test
data.

## Success criteria

- Tiny-overfit candidate/action/evidence accuracy at least 0.95.
- ID final-decision accuracy at least 0.80.
- Zero deterministic replay mismatches.
- Zero row-permutation decision mismatches within FP32 tolerance.
- A recurrent Spider primary score above the pooled baseline.

Failure to beat the pooled baseline is a valid result and must be reported.

## Constraints

- Maximum 12 accepted architecture/hyperparameter experiments.
- Current initial loop budget: 8 accepted experiments.
- One exploration seed per accepted search experiment.
- Three independent seeds for the best two frozen configurations.
- Five-minute timeout per exploration experiment.
- Parameter budget: 25,000--2,000,000 trainable parameters.
- No custom CUDA/Triton kernels.
- No natural-language encoder or learned manifold writer.
- No generator enum, node ID, relation ID, path label, or answer label in model
  inputs.
- No optimisation on the sealed test.
- No silently discarded negative result.

## Search space

Pre-registered comparisons:

1. pooled symmetric baseline;
2. flat position-free Transformer baseline;
3. tied recurrent multi-set Spider with standard cross-attention;
4. tied recurrent multi-set Spider with compositional edge attention;
5. tied versus untied recurrence;
6. one versus two hypotheses per destination;
7. with versus without global evidence;
8. with versus without equivalent-view consistency;
9. context value-of-information with versus without read cost;
10. functional swap augmentation/consistency as an optional final ablation.

Permitted hyperparameters:

- learning rate: `1e-4` to `5e-3`;
- `d_model`: 32, 64, or 128;
- heads: 2 or 4 with divisible head dimensions;
- tied blocks: 1 or 2;
- training rounds: 4, 6, or 8;
- frontier width: 16, 32, or 64;
- hypotheses per destination: 1, 2, or 4;
- context reads: 0, 2, 4, or 8;
- loss weights already exposed by `SpiderLossConfig`.

## Iteration protocol

For each hypothesis:

1. state one falsifiable expected metric change;
2. change one primary variable or one explicitly coupled mechanism;
3. run the mechanical evaluator under `timeout 5m`;
4. compare against the current baseline and guard metrics;
5. keep a successful change, or revert an unsuccessful code-only mutation;
6. append JSONL and Markdown records regardless of outcome;
7. update `artifacts/spider_v0/progress.svg`.

The loop does not pause for approval (`pause_every: never`) and stops at the
accepted-experiment budget or when no remaining pre-registered comparison is
informative.

## Pre-registered hypotheses

| ID | Comparison | Falsifiable expectation |
|---|---|---|
| H01 | recurrent standard vs pooled | higher path-length OOD valid-path rate |
| H02 | recurrent standard vs flat | higher equivalent-view OOD accuracy at matched budget |
| H03 | compositional vs standard | higher path-length OOD recall without transform collapse |
| H04 | tied vs untied | tied model loses no more than 0.03 ID accuracy and improves longer-path OOD |
| H05 | two vs one hypotheses | higher conflict accuracy with bounded search-cost increase |
| H06 | evidence vs no evidence | higher corroboration/conflict exact evidence accuracy |
| H07 | view consistency vs none | lower equivalent-view mismatch without ID collapse |
| H08 | context VOI/read cost vs none | fewer unnecessary reads at equal useful-read recall |
| H09 | functional swapping | improved swap OOD or a documented collapse/no-effect result |

## Guard tests

```bash
.venv/bin/pytest -q
```

CUDA tests run whenever `torch.cuda.is_available()` is true. A CUDA
experiment additionally records device name, PyTorch version, runtime CUDA
version, dtype, and peak allocated memory.

## Experiment history

No experiments have run yet. The runner appends a generated table here and to
`artifacts/spider_v0/EXPERIMENT_SUMMARY.md`.
