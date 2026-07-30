# Spider v0.3: Evidence Recovery and Autonomous Control

## Question

Spider v0.3 asks whether the pooled recurrent Spider can become a reliable
autonomous controller after evidence coverage, ranking, calibration, and
termination are measured and supervised at the controller-state level.

The historical v0.2 result remains immutable. This milestone uses only
non-sealed development data and records negative results.

## Evidence identity and stages

Evidence is edge-specific when a program declares `evidence_edge_ids`;
otherwise it is node-specific. This distinction is necessary for converging
paths whose final candidates share a destination but have different semantic
validity.

Every required item is followed through:

1. structural reachability under remaining controller budgets;
2. full CSR enumeration;
3. budget-sliced scoring;
4. evidence-policy selection;
5. exact-ledger recording.

Evidence selection is independent of frontier expansion. A terminal candidate
may be recorded without becoming a next-round hypothesis.

## Registered evidence comparison

The primary data are the existing non-sealed four-family development cases.
Validation is divided by base-case group into disjoint calibration and
evaluation roles. The recurrence-necessity data remain a long-horizon
diagnostic.

E0 preserves the current evidence objective. E1 replaces its set term with
class-weighted BCE and a multi-positive hard-negative ranking loss. E2 adds
structurally plausible invalid candidates. Focal loss and unrestricted top-k
selection are excluded.

## Termination

The factorized head receives direct state targets for evidence sufficiency,
useful reachable work, answer support, and unknown reason. These labels may
not be reconstructed from a final six-way class.

NULL is a per-hypothesis expansion action. Selecting it removes only that
branch; it neither erases other branches nor globally terminates execution.

## Conditional architecture test

Only a successful pooled controller unlocks the multi-binding comparison.
That benchmark delays the selector until after an opaque key/value map has
been consumed and makes projected mean/max statistics identical across cases
with different answers. This tests a use case where row-level interactions,
rather than model size or stopping behaviour, are the controlled difference.
