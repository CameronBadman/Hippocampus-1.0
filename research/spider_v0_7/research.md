# Spider v0.7 canonical binding AutoResearch

## Goal

Determine whether explicit training-only cross-modal co-reference supervision
can recover exact evidence for unseen symbols while preserving Spider v0.6's
high precision. This campaign isolates representation alignment and evidence
readout; it does not change traversal capacity, termination, or the manifold
substrate.

## Success metric

Primary score: the minimum of development exact evidence-set accuracy,
evidence precision, and evidence recall. Higher is better.

Target: score at least 0.82, with precision at least 0.90 and scored-positive
coverage at least 0.98. The inference rule is frozen as
`candidate_energy > null_energy`; no temperature, threshold, cardinality, or
family-specific policy is fitted after training.

Additional advancement gates:

- unseen-symbol binding AUROC at least 0.99 and Top-1@256 at least 0.95;
- lookup average precision at least 0.95;
- lookup recall improves by at least 0.50 on two of three matched seeds;
- exact evidence-set accuracy improves by at least 0.05;
- latest-valid and corroboration recall regress by no more than 0.01;
- deterministic replay and row-permutation mismatch counts remain zero.

## Constraints

- Evaluator: `.venv/bin/python scripts/run_spider_v0_7_autoresearch.py`
- Keep policy: `pass_only`
- `pause_every: never`
- `max_iterations: 3`
- `noise_runs: 3`
- `min_delta: 0.01`
- Historical freeze: tag `spider-v0.6-zero-shot`, commit `b4e8aa1`.
- Matched seeds: 1701, 1802, and 1903.
- Development accelerator: one local NVIDIA GeForce RTX 5070 Ti, FP32.
- Screen length: 1,000 updates; checkpoints every 250 updates.
- Finalist confirmation: 2,000 updates only after the screen gate passes.
- Evaluation symbols must be absent from training and model selection.
- Co-reference equality may supervise the training loss but is never a model
  feature or an inference input.
- Renderer inverse matrices, AST fields, oracle fields, program IDs, relation
  IDs, node IDs, and row positions are forbidden model inputs.
- Guards: all tests pass; finite metrics; zero symbol overlap; zero sealed
  accesses; zero deterministic replay mismatches; zero row-permutation decision
  mismatches.
- Forbidden changes: larger models, learned termination, compositional edges,
  natural-language writers, calibrated inference, and sealed evaluation.

## Frozen benchmark correction

The historical lookup generator gave the requested value to every foreground
destination. Spider v0.7 replaces it with matched four-way candidates:

1. requested relation, requested value, valid gate;
2. requested relation, wrong value, valid gate;
3. wrong relation, requested value, valid gate;
4. requested relation, requested value, invalid gate.

Matched no-positive cases retain the same symbol inventory, cardinalities,
degrees, and scalar inventory while breaking the only valid conjunction. All
other program-family semantics remain unchanged. The resulting dataset is a
new version and never rewrites v0.6 manifests or results.

## Current approach

Spider v0.6 Z1 is the high-precision parent: precision 0.9495, recall 0.7608,
and exact-set accuracy 0.7910. Scored-positive coverage is 0.9992, but lookup
recall is zero. Raw renderer vectors support perfect unseen-symbol retrieval,
whereas direct retrieval after the learned family projections is near chance.
Prior dedicated, slot-aware, and pairwise heads did not receive an explicit
alignment objective and did not recover lookup.

Spider v0.7 introduces a dedicated evidence-local canonical space. Separate
query, edge, and summary projections are trained with symmetric multi-positive
InfoNCE over naturally co-referent observable rows. Candidate evidence scoring
uses permutation-invariant query-to-edge and query-to-destination match
statistics plus the unchanged path and controller state.

## Search space

1. R0: frozen Z1 backbone with the ordinary evidence head and candidate NULL
   retrained on the corrected benchmark.
2. R1: replace only the evidence head with a canonical pairwise binding head,
   without alignment supervision.
3. R2: R1 plus training-only cross-modal InfoNCE, temperature 0.07 and weight
   0.1.

If R2 passes the screen, repeat R0 and R2 as full end-to-end 2,000-step runs.
No additional arm is permitted in this campaign.

## History

| Iteration | Hypothesis | Status | Score | Notes |
| ---: | --- | --- | ---: | --- |
| 0 | Freeze the corrected binding protocol. | kept | 0.7608 | Historical Z1 score; new benchmark control pending. |

