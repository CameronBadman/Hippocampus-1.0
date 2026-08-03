# Spider v0.5 Development Dataset Card

## Identity

- Version: `spider-programs-v0.5-score-decode-dev`
- Renderer: `renderer-v0.4`, fixed orthogonal-aligned geometry
- Aggregate hash: `b3de9d584148859f4f12a17377005c969372de753677e2b3360fa5e2fa7ada68`

This dataset exists only to isolate evidence scoring from evidence-set decoding.
It does not replace or reinterpret any Spider v0 through v0.4 result.

## Partitions

| Partition | Base cases | Purpose |
| --- | ---: | --- |
| Training | 8,192 | parameter learning |
| Model selection | 512 | checkpoint selection |
| Calibration | 512 | temperature and operating policy |
| Development evaluation | 1,024 | one-time arm evaluation |

Partitions are balanced by program family, answerable/unknown outcome,
required evidence cardinality, path length, and graph size. Related equivalent
views stay in one partition. The generator uses fresh partition seeds, and
training applies fresh manifold-row permutations on presentation.

## Program families

The four retained families are lookup, reachability, latest-valid, and
corroboration. Their semantics and the aligned A2 observation renderer are held
fixed so the experiment changes only evidence scoring and set selection.

No relation IDs, node IDs, row positions, oracle costs, answer labels, path
membership labels, or latent program enums are supplied as neural features.

## Leakage and access policy

The manifests are deterministic and disjoint. Query-cardinality metadata-only
accuracy is 0.5, so trivial answerability metadata does not solve the split.
The campaign materialises no sealed partition. All twelve accepted records
report `sealed_access_count: 0`.

The development-evaluation split was opened once per frozen selected
checkpoint, as registered. It must not now be reused to select another v0.5
architecture or operating threshold.
