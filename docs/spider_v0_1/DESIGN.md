# Spider v0.1 frozen design

## Scientific scope

Spider v0.1 asks one question:

> Can the existing tied recurrent multi-set Spider outperform its pooled
> control in autonomous graph execution after controller/training mismatch is
> removed?

The renderer, packed manifold substrate, family widths, standard edge
attention, tied recurrence, model width, path/evidence sizes, and four program
families remain fixed. Compositional attention, functional swapping, learned
writers, language inputs, and reinforcement learning are excluded.

## One transition state machine

Every execution mode uses the following sequence:

```text
propose packed arcs and score with aligned controls
  -> label the actual state
  -> choose context actions
  -> gather/refine selected contexts
  -> choose evidence and frontier actions independently
  -> update neural evidence and exact evidence ledger
  -> construct next hypotheses and trace ledger
  -> build post-transition termination controls
  -> supervise and execute termination
```

The public records are:

- `ControllerProposal`: packed `FrontierExpansion`, pre-context outputs,
  candidate controls, depth-valid mask, and resolved limits;
- `ControllerActions`: independent frontier, context, and evidence indices
  plus the source of each discrete action;
- `ControllerTransition`: refined outputs, next hypotheses/evidence/state,
  ledgers, and post-transition termination controls;
- `ControllerResult`: autonomous decision, exact traces, action-source
  diagnostics, final logits, and efficiency counts.

`SparseWavefrontController.propose` is the only candidate scoring path.
`SparseWavefrontController.apply` is the only state mutation path. Oracle,
scheduled, autonomous, and replay execution differ only in the action policy.

## Pure controller features

Candidate controls have width six:

1. parent depth / configured maximum depth;
2. current round / configured maximum rounds;
3. remaining search budget / resolved search limit;
4. remaining context budget / resolved context limit;
5. search budget exhausted;
6. context budget exhausted.

Termination controls have width six:

1. completed rounds / configured maximum rounds;
2. used search budget / resolved search limit;
3. used context budget / resolved context limit;
4. next frontier empty;
5. search budget exhausted;
6. context budget exhausted.

Zero-capacity budgets use a denominator of one and set their exhaustion flag.
No feature depends on oracle trace length.

## State oracle

`StateOracle` is supervisor-only. It converts packed IDs to case-local IDs and
labels actual candidate occurrences using the union of valid program
transitions, exact evidence requirements, contexts already read, trace
references, and remaining budgets.

A state is recoverable when at least one current hypothesis can still reach a
missing required evidence node through program-valid transitions within depth,
round, and search limits. Exact frontier equality is never required.
Duplicates preserve duplicate targets.

Post-transition termination is:

- `ANSWER` when required answer evidence has been accumulated;
- `CONTINUE` while a valid completion remains within budget;
- `UNKNOWN_ABSENT` when search is complete and no support exists;
- `UNKNOWN_CONFLICT` when exact conflicting evidence is accumulated or
  irreconcilably reachable;
- `UNKNOWN_INCOMPLETE` when a valid completion exists but remaining controller
  capacity cannot reach it;
- `UNKNOWN_UNSUPPORTED` for unsupported query surfaces.

## Scheduled actions

Four teacher-forcing fractions are sampled independently per round:

- frontier;
- context;
- evidence;
- termination execution.

Every encountered state is supervised regardless of the executed source.
Diagnostics record each source and selected indices. The default five phases
are `(1,1,1,1)`, `(0.75,0.75,0.75,1)`,
`(0.5,0.5,0.5,0.75)`, `(0.25,0.25,0.25,0.5)`, and
`(0,0,0,0)`.

## Sparse decisions

Context, evidence, and frontier actions are independent. Context refinement
happens before evidence/frontier decisions. Evidence updates happen before
termination and do not require frontier retention.

The model frontier first filters depth-invalid candidates and candidates below
the configured expand threshold, then applies stable priority order, the
per-destination hypothesis cap, and global frontier cap. Arc ID remains the
final snapshot-local tie-breaker.

## Termination heads

The control model retains the flat six-way head. The registered ablation uses:

1. continue versus stop;
2. answer versus unknown, on stop targets;
3. absent/conflict/incomplete/unsupported, on unknown targets.

Hierarchical probabilities are composed into the same six decisions for the
controller. Losses and confusion matrices retain the hierarchy.

## Evidence objective and calibration

The registered evidence objective combines:

- class-balanced candidate BCE;
- positive listwise evidence mass;
- soft positive recall;
- a modest false-positive probability penalty.

Counts, precision/recall curve, average precision, exact set accuracy, and
conditional recalls are reported. An operating threshold is selected by
maximising development evidence F1 with deterministic lower-threshold
tie-breaking. Calibration rejects sealed/test split names.

## Revision rule

This design is frozen before experiments. It changes only for a demonstrated
correctness defect or an implemented substrate API conflict, and every change
must be recorded in the final report.
