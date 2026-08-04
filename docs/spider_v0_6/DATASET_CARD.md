# Spider v0.6 Development Dataset Card

## Purpose

`spider-programs-v0.6-zero-shot-dev` tests evidence-set selection when every
observable surface symbol at evaluation is absent from training. It retains
the four synthetic program families and the aligned orthogonal renderer from
v0.4. It contains no natural-language data and is not a production benchmark.

## Partitions

| Partition | Cases | Observable symbols | Manifest SHA-256 |
| --- | ---: | ---: | --- |
| Training | 8,192 | 667,648 | `6f8e15f83c5154d201281f3569fd3de77a122161e5a1e2829e42a632e51f1193` |
| Model selection | 512 | 41,728 | `99aaef837131533b0a8879e8e3a6e46092dd5be331c9124b5c929d3a38379966` |
| Diagnostic calibration | 512 | 41,728 | `d2f331cc68df3c48b9f5ed1623f05267cb3c4cab98ac08aff450c629d2e7808b` |
| Development evaluation | 1,024 | 83,456 | `15a2e75c5a486db711bedac3663e303f939d437f30bca612d8376a10cc9c8188` |

The aggregate manifest hash is
`a05d301bef77d86abcbb658444c2ed277545b82653dac202cef319ecfc1abc17`.
All 834,560 observable symbols are partition-local; measured overlap is zero.
Related views of a base case stay in one partition.

Each partition is balanced across lookup, reachability, latest-valid, and
corroboration; answerable and unknown outcomes; graph sizes 8, 16, 24, and 32;
and required evidence cardinalities. Query-cardinality answerability accuracy
is 0.5, providing a basic metadata-leakage check.

## Renderer and zero-shot boundary

The frozen `renderer-v0.4` uses shared latent symbol identity followed by
seeded, fixed orthogonal modality transforms. Fresh row permutations are used
during training. The evaluation model sees no partition-external lookup table,
symbol ID, program enum, node ID, row position, oracle label, fitted
temperature, fitted threshold, or fitted evidence count.

The directory named `calibration` is retained for protocol compatibility. In
v0.6 it is diagnostic only: all accepted runs record temperature 1.0 and no
fitted operating policy.

## Access policy and limitations

No sealed split was generated, materialised, or evaluated. Development
evaluation was opened once per selected checkpoint, so it must not become a
future model-selection partition. Synthetic codebook generalisation does not
establish natural-language, real-world graph, or production retrieval quality.
