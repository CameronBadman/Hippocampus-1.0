# Spider v0.7 design: canonical cross-modal binding

## Decision

Spider v0.7 tests one hypothesis: task-only evidence supervision does not teach
the evidence pathway to canonicalise the renderer's fixed modality transforms.
Training-only co-reference supervision should make relation and value matches
transfer to entirely unseen symbols.

This remains zero-shot with respect to evaluation symbols. Equality of raw
training symbols creates loss targets only. Symbol strings, renderer matrices,
program-family labels, and oracle fields never enter the neural forward path.

## Representation ladder

A frozen-checkpoint audit measures same-symbol retrieval and candidate binding
at raw renderer rows, family-projected rows, symmetric family pools, proposed
path rows, mean path state, and final evidence-minus-NULL energy. Causal swaps
of relation, destination value, and gate test whether each stage responds to
the intended observation.

## Evidence-local canonical space

The canonical binding readout receives raw packed query, edge, and destination
summary sets alongside the existing path and controller features. Independent
linear canonicalisers map rows into one normalised space. Symmetric maximum and
log-mean-exp similarities expose query-to-edge and query-to-destination
matches; their minimum and product expose the required conjunction. A
dedicated MLP produces the candidate evidence energy.

The alignment objective is symmetric multi-positive InfoNCE over observable
row pairs sharing at least one symbol. Rows with no shared symbol are in-batch
negatives. Scalar-only rows do not create positive pairs. Pair targets are
supervisor-side data aligned through row permutation and are unavailable at
inference.

## Controller boundary

The candidate-conditioned NULL policy from Spider v0.6 Z1 remains fixed in
form and is retrained with the evidence head because the logit scale changes.
Selection uses the registered raw comparison `candidate_energy > null_energy`.
Evidence recording remains independent of frontier expansion.

No packed-graph, topology, controller-transition, traversal, termination, or
model-capacity changes are in scope.
