# Spider v0.3 Dataset Card

## Purpose

Spider v0.3 reuses the non-sealed `spider-programs-v0.2` development
semantics to isolate evidence selection and termination. It does not introduce
new language writers, relation labels, or architecture cues. The benchmark
still contains lookup, reachability, latest-valid, and corroboration cases.

This milestone does not open any Spider v0 or v0.2 sealed split.

## Registered partitions

| Partition | Cases | Role |
|---|---:|---|
| train | 512 | Evidence and controller-state training |
| development calibration | 64 | Temperature and one global evidence threshold |
| development evaluation | 64 | Frozen-threshold evidence and controller evaluation |

The aggregate manifest SHA-256 is
`0ed8e27ec44f3773f76b79f1947526f33ba233556b7db91fef04dcb647e5409d`.
Calibration and evaluation are grouped by base-case identity, so related views
cannot occur on both sides of the threshold-selection boundary.

The training split is exactly balanced across the four program families at
128 cases each. Calibration contains 17 corroboration, 16 latest-valid, 16
lookup, and 15 reachability cases. Evaluation contains 15 corroboration, 16
latest-valid, 16 lookup, and 17 reachability cases.

## Evidence supervision

Evidence identity is edge-specific whenever a case declares exact evidence
edges. It is otherwise node-specific. This prevents converging valid and
invalid arcs into the same destination from being treated as the same
evidence action.

Each required item is measured at five stages:

1. reachable under the remaining depth, round, and search budgets;
2. enumerated by packed CSR frontier expansion;
3. scored before the evidence policy is applied;
4. selected by the calibrated evidence policy;
5. recorded in the exact evidence ledger.

The renderer and graph-program oracle remain separate. Oracle labels, internal
program families, node IDs, answer locations, and path membership are not
model inputs.

## Threshold policy

Temperature and threshold selection use only `development_calibration`.
`development_evaluation` is read after the operating point is fixed. The
objective is exact evidence-set accuracy under the registered precision
constraint. The controller uses one global threshold; it receives neither
required evidence cardinality nor an unrestricted top-k oracle.

## Limitations

This is a small synthetic development benchmark. Its 64-case calibration and
evaluation partitions make seed-level differences coarse. The results test
controller mechanics and exchangeable manifold processing, not natural
language understanding or production retrieval.

E2 received no additional structurally plausible negative targets on the
observed development rollouts, so its learned parameters were byte-identical
to E1. E2 is therefore a recorded non-informative arm, not independent
evidence about hard-negative mining.
