# Spider v0.5 score-versus-decode AutoResearch protocol

## Goal

Determine whether Spider v0.4 evidence failures come from (a) missing explicit
cross-manifold identity matching in the candidate scorer, (b) a set decoder
that cannot observe the candidates it must select, or both. Improve recall
without purchasing it through false-positive evidence.

## Success metric

Primary: development exact evidence-set accuracy, subject to evidence precision
at least 0.90 and scored-positive coverage at least 0.98.

Required advancement deltas over the matched X0 control are:

- overall exact evidence-set accuracy: at least +0.05;
- overall evidence recall: at least +0.05;
- lookup evidence recall: at least +0.20;
- no latest-valid or corroboration exact-set regression greater than 0.02.

Tie-breakers are macro average precision, fewer false positives per case, lower
cardinality error, better worst-positive rank, and earlier checkpoint.

## Constraints

- Evaluator: `.venv/bin/python scripts/run_spider_v0_5_autoresearch.py`
- Keep policy: `score_improvement`
- `pause_every: never`
- `max_iterations: 4`
- `noise_runs: 3`
- `min_delta: 0.02`
- Baseline source/config freeze: commit `92b3ea7`, Spider v0.4 D0/F0.
- Seeds: 1701, 1802, and 1903.
- Development accelerator: one local NVIDIA GeForce RTX 5070 Ti, FP32.
- Maximum steps: 2,000; checkpoints every 250 steps.
- The renderer is v0.4 fixed orthogonal-aligned A2.
- Traversal, fixed-horizon execution, evidence updating, controller budgets,
  and non-evidence policy heads remain unchanged.
- X0 is the existing evidence scorer with calibrated global threshold.
- X1 changes only the evidence scorer to an exchangeable pairwise matcher.
- X2 changes only selection to a current-candidate-set count decoder.
- X3 combines X1 and X2 to measure interaction.
- Candidate-count supervision describes unique, not-yet-recorded required
  evidence actions visible in the current candidate set. It never describes
  total case cardinality.
- Guards: finite metrics; precision at least 0.90; scored-positive coverage at
  least 0.98; zero deterministic replay mismatches; zero row-permutation
  decision mismatches; all tests passing; zero sealed accesses.
- Forbidden changes: larger models, additional path rows or blocks,
  compositional edges, learned termination, natural-language writers,
  relation/type IDs, row positions, generator labels as features, and access
  to any sealed split.

## Current approach

The retained v0.4 F0 pooled control has 0.7549 exact evidence-set accuracy,
0.8830 precision, 0.7516 recall, and 0.9992 scored-positive coverage. Its
overall oracle-cardinality ceiling is 0.9274 exact set. The ceiling is not
uniform: reachability ranking is strong under oracle cardinality, while lookup
ranking remains weak.

The failed v0.4 learned cardinality head consumes pooled query, evidence,
frontier, and controller state, predicts total required evidence, then subtracts
the accumulated count. It does not consume the current candidates. This
campaign corrects that action/observation mismatch and separately tests an
explicit cross-manifold identity matcher.

## Search space

| Arm | Candidate evidence score | Evidence selection |
| --- | --- | --- |
| X0 | v0.4 shared pooled head | calibrated global threshold |
| X1 | pairwise query/edge/destination matcher | calibrated global threshold |
| X2 | v0.4 shared pooled head | current-candidate-set count decoder |
| X3 | pairwise query/edge/destination matcher | current-candidate-set count decoder |

The pairwise matcher uses learned shared-width projections, masked pairwise
similarities, and symmetric max/log-mean-exp aggregation. It receives no row
positions or identifiers. The candidate decoder uses a permutation-invariant
set encoder over current candidate states and logits and predicts `0,1,2,3,4+`
currently selectable evidence actions.

## Execution protocol

1. Verify X0 on the new non-sealed development partitions.
2. Treat checkpoints through step 1,000 of seed 1701 as the screen; do not run
   a duplicate short training job.
3. Continue the same registered 2,000-step training schedule and complete all
   three matched seeds for the factorial unless a mechanical guard fails.
4. Select checkpoints only on model selection.
5. Fit threshold/operating policy only on calibration.
6. Evaluate development evaluation once per selected checkpoint.
7. Preserve all negative results and stop after the matrix is exhausted or an
   arm passes the full advancement gate.

## History

| Iteration | Hypothesis | Status | Score | Notes |
| ---: | --- | --- | ---: | --- |
| 0 | Freeze the score-versus-decode factorial before implementation. | kept | 0.7549 | v0.4 remains immutable; no sealed data is in scope. |
| 1 | Re-establish the shared-head/global-threshold control on the new partitions. | kept | 0.7754 | X0: precision 0.9444, recall 0.7444; lookup recall is zero on every seed. |
| 2 | Explicit pairwise cross-manifold matching fixes opaque evidence ranking. | discarded | 0.7793 | X1: precision 0.9313, recall 0.7584; no lookup recovery and zero gate wins. |
| 3 | A decoder that observes the current candidates fixes threshold under-selection. | discarded | 0.6761 | X2: raw exact set 0.8350 and recall 0.9075, but precision 0.7411 fails the guard. |
| 4 | Pairwise matching makes current-candidate count selection precise. | discarded | 0.7220 | X3: raw exact set 0.8252, precision 0.8130, and unstable seed behavior. |

The four-iteration budget is exhausted. No treatment advances, X0 remains the
finalist, and no sealed or A100 stage opens.
