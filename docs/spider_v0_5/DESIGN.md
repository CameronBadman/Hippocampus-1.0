# Spider v0.5: Candidate Scoring and Evidence-Set Decoding

Spider v0.5 isolates two different failures visible in the v0.4 evidence
results. Lookup candidates are not reliably ranked even when evidence
cardinality is supplied by an oracle. Reachability candidates are ranked well,
but a global threshold misses many positives. One mechanism cannot explain
both observations.

## Pairwise evidence matcher

The matcher augments only the evidence score. It computes masked, learned
pairwise similarities for query-to-edge, query-to-destination, and
edge-to-destination manifold rows. Each similarity matrix is reduced with
position-free maximum and log-mean-exp statistics. These statistics, the
candidate path state, and controller features feed an evidence-only MLP.

No row positions, node IDs, edge IDs, relation IDs, or latent program labels
enter the neural computation. Row permutation therefore changes only reduction
order, not represented content.

## Current-candidate-set decoder

The decoder operates on the candidates available in one graph and controller
round. Candidate tokens contain the proposed path state and learned policy
outputs. A symmetric set encoder aggregates them and predicts the number of
unique outstanding evidence actions currently present: zero, one, two, three,
or four-or-more.

This differs deliberately from v0.4. The historical head predicted total case
evidence cardinality from pooled global state and subtracted previously selected
evidence. That policy could force false positives before required evidence was
visible. The v0.5 zero class is also the explicit no-evidence action for the
current set.

Context refinement occurs before the final decoder prediction. Evidence
selection remains independent from frontier expansion.

## Factorial comparison

| Arm | Matcher | Candidate-set decoder |
| --- | --- | --- |
| X0 | no | no |
| X1 | yes | no |
| X2 | no | yes |
| X3 | yes | yes |

All other model, renderer, controller, horizon, loss, and data settings are
matched. The campaign is development-only and does not materialise or evaluate
any sealed split.
