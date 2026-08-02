# Renderer identifiability

Spider v0.4 first tests whether observable identity can be recovered across
query, edge, summary, and context modalities. This is a prerequisite for a
meaningful graph-model comparison: no model should be expected to generalise a
fresh opaque symbol when every modality assigns it an unrelated hash vector.

## Registered arms

- **A0, independent:** the historical `hash(modality, symbol)` geometry.
- **A1, shared additive:** a common symbol latent plus a modality embedding.
- **A2, orthogonal aligned:** a common symbol latent transformed by a fixed,
  seeded orthogonal matrix for each modality, plus a modality embedding.

A small two-projection bilinear probe was trained on one symbol set and tested
on unseen symbols. The test covered query-to-edge, query-to-summary, and
edge-to-summary matching, including retrieval among 64 and 256 distractors.
No graph model was trained in this phase.

## Result

| Arm | Minimum AUROC | Minimum Top-1@64 | Minimum Top-1@256 | Decision |
|---|---:|---:|---:|---|
| A0 | 0.4780 | 0.0146 | 0.0029 | historical negative control |
| A1 | 1.0000 | 1.0000 | 1.0000 | upper bound passes |
| A2 | 1.0000 | 1.0000 | 1.0000 | gate passes |

The preregistered A2 gate was AUROC >= 0.99, Top-1@64 >= 0.95, and
Top-1@256 >= 0.85. A2 clears all three requirements while preserving distinct
modality coordinates. Row permutation checks also passed exactly.

This result establishes identifiability of the synthetic interface, not graph
reasoning ability. The next causal test holds the pooled model and evidence
objective fixed while changing only renderer geometry.

Machine-readable results live under
`artifacts/spider_v0_4/renderer_identifiability/`. No sealed split was accessed.
