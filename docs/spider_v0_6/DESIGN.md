# Spider v0.6: Zero-Shot Candidate-Relative Evidence Energy

## Meaning of zero-shot

Spider v0.6 is trained on synthetic graph programs, but inference is zero-shot
with respect to surface identity and operating policy:

- every evaluation symbol is absent from training;
- model weights are frozen before development evaluation;
- no temperature, probability threshold, evidence count, or family-specific
  policy is fitted after training; and
- the same raw decision rule is used for every seed, program family, round,
  and graph.

This is structural and symbol zero-shot generalisation. It is not a claim that
an untrained network can solve graph programs.

## Energy selector

For each graph and controller round, the model already produces one evidence
energy for every expanded candidate. A new permutation-invariant set encoder
reads:

- the proposed path-state rows pooled symmetrically per candidate;
- all seven candidate policy outputs;
- pooled query, accumulated evidence, and active-frontier state; and
- the canonical controller features.

It emits one graph-local NULL energy. Candidate `i` is evidence exactly when:

```text
evidence_energy[i] - null_energy[graph(i)] > 0
```

The zero margin is immutable. Multiple candidates may exceed NULL, and all may
remain below it. Evidence inclusion is still independent from frontier
expansion.

## Why this differs from prior NULL policies

The v0.4 global NULL head did not observe the candidate set. The v0.5 count
decoder observed candidates but forced a number of selections and accumulated
false positives across rounds. The v0.6 head observes both the candidate set
and global evidence state, then learns a relative boundary without predicting
cardinality.

Rows and candidates remain exchangeable. Mean/max segmented reductions are
used only to construct the set-level NULL energy; no insertion order, physical
row position, node ID, edge ID, program family, or oracle field is a feature.

## Frozen scope

The pooled recurrent processor, aligned orthogonal renderer, dimensions,
controller, fixed horizon, optimizer, and graph substrate are unchanged. The
campaign does not test a larger Spider, learned stopping, natural-language
writers, or sealed evaluation.

## Registered training variants

- Z0: the historical global learned NULL, with no post-training calibration.
- Z1: candidate-conditioned NULL with candidate-weighted relative BCE.
- Z2: Z1 with positive and negative relative losses balanced within each
  graph state.
- Z3: Z2 plus a weight-0.25, margin-0.2 loss over all positives and at most
  four highest-scoring structurally plausible negatives per graph.

Only Z1 changes the selector architecture. Z2 and Z3 change training losses;
all four use the same immutable zero-margin inference rule. The completed
results are in [FINAL_REPORT.md](FINAL_REPORT.md).
