# Spider v0.6 zero-shot evidence energy AutoResearch

## Goal

Rebuild evidence selection so an unchanged model can recover exact evidence
sets for wholly unseen surface symbols without fitting a temperature, threshold,
or cardinality policy after training. The selector must be able to choose no
evidence from the current candidate set.

## Success metric

Primary score: the minimum of development exact evidence-set accuracy,
evidence precision, and evidence recall. Higher is better.

Target: score at least 0.82, with scored-positive coverage at least 0.98.
All metrics are computed on symbol-disjoint development cases using the raw
zero-margin decision `candidate_energy > null_energy`.

Tie-breakers are exact evidence-set accuracy, macro average precision, fewer
false positives per case, lower cardinality error, and earlier checkpoint.

## Constraints

- Evaluator: `.venv/bin/python scripts/run_spider_v0_6_autoresearch.py`
- Keep policy: `score_improvement`
- `pause_every: never`
- `max_iterations: 4`
- `noise_runs: 3`
- `min_delta: 0.01`
- Baseline source freeze: commit `3bcf9c3`.
- Matched seeds: 1701, 1802, and 1903.
- Development accelerator: one local NVIDIA GeForce RTX 5070 Ti, FP32.
- Maximum training steps: 2,000; checkpoints every 250 steps.
- Training, model-selection, diagnostic, and development-evaluation symbol
  namespaces must be pairwise disjoint.
- Model selection may select weights but may not fit any inference-time scalar
  or policy on held-out data.
- The inference rule is fixed before training and contains no calibrated
  threshold, oracle evidence count, family ID, node ID, relation ID, or row
  position.
- Traversal, renderer geometry, fixed-horizon execution, controller budgets,
  model width, and non-evidence heads remain fixed.
- Guards: all tests pass; finite metrics; coverage at least 0.98; zero symbol
  overlap; zero deterministic replay mismatches; zero row-permutation decision
  mismatches; zero sealed accesses.
- Forbidden changes: larger models, learned termination, compositional edges,
  natural-language writers, sealed data, relation/type IDs, generator labels
  as features, and post-training threshold or temperature fitting.

## Current approach

Spider v0.5 X0 reaches 0.9444 precision but only 0.7444 recall using a
per-seed calibrated threshold. X2 proves that candidate ranking contains more
signal: recall reaches 0.9075, but a learned count policy reduces precision to
0.7411. The useful information is therefore present, while the decision to
select nothing is unreliable.

The historical learned-null head reads pooled query, frontier, and evidence
state before observing the candidates it must reject. v0.6 instead defines a
candidate-conditioned energy model. A permutation-invariant encoder observes
the current candidate set and accumulated neural evidence, emits one graph-local
NULL energy, and compares every candidate evidence energy directly to it.

## Search space

1. Re-establish the existing global learned-null policy without calibration.
2. Replace it with a current-candidate-set NULL energy.
3. Balance the relative-energy objective by graph and zero/nonzero candidate
   state if plain candidate BCE overweights large negative sets.
4. Add a bounded hard-negative margin only if candidates remain incorrectly
   ordered around the learned NULL boundary.

Only one change is introduced per iteration. Failed variants are preserved in
the ledger and reverted before the next hypothesis.

## History

| Iteration | Hypothesis | Status | Score | Notes |
| ---: | --- | --- | ---: | --- |
| 0 | Freeze the symbol-disjoint, calibration-free protocol. | kept | 0.7444 | v0.5 recall is the limiting control statistic; no sealed data is in scope. |
| 1 | Re-run the global learned NULL without calibration. | control | 0.7600 | Three seeds; exact 0.7809, precision 0.9322, recall 0.7600. Lookup recall was zero. |
| 2 | Condition NULL on the current candidate set. | rejected | 0.7608 | Exact set improved by 0.0101, but recall improved by only 0.0008 and the arm won one of three seeds. |
| 3 | Balance relative-energy BCE within each graph state. | kept parent | 0.7998 | Won two seeds and raised recall to 0.8021, but precision fell to 0.8722 and lookup recall stayed at 0.0026. |

Iteration 4 is therefore the final pre-registered bounded hard-negative margin
around NULL, layered on the graph-balanced objective. The architecture, data,
raw zero-margin inference rule, and all non-evidence losses remain fixed.
