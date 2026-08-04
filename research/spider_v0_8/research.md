# Spider v0.8 SRE graph-retrieval transfer

## Goal

Determine whether the canonical alignment principle validated by Spider v0.7
transfers to entity-disjoint, natural-language SRE incident-memory retrieval
when execution uses the packed graph substrate.

This is a development-only transfer experiment. It does not reopen any Spider
sealed split or the SRE Incident-Memory sealed test.

## Success metric

Primary score is the harmonic mean of development-evaluation:

- answerable-case MRR;
- Recall@8;
- macro evidence average precision; and
- macro hard-negative pairwise accuracy.

The target is a score of at least 0.85, with each component at least 0.80.
Exact evidence-set accuracy, candidate-NULL precision/recall, false positives,
latency, deterministic replay, and row-permutation invariance are guards and
diagnostics rather than substitutes for ranking quality.

## Constraints

- Evaluator: `.venv/bin/python scripts/run_spider_v0_8_autoresearch.py`
- Keep policy: `pass_only`
- `pause_every: never`
- `max_iterations: 3`
- `noise_runs: 3`
- `min_delta: 0.01`
- Source corpus: SRE Incident-Memory v3, read-only sibling checkout.
- Train: 1,600 cases.
- Validation is deterministically divided into 100 model-selection and 100
  development-evaluation cases, balanced by observable scenario family and
  required-evidence cardinality.
- Public demo: three independent demo-only cases; never a metric split.
- The SRE `test.*` files are forbidden and transfer commands reject them.
- Frozen text encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Development device: one local NVIDIA GeForce RTX 5070 Ti, FP32.
- Seeds: 1701, 1802, and 1903 for confirmation.
- No candidate IDs, relevance labels, adversary labels, relationship paths,
  generator fields, or scenario-family labels enter model inference.
- No fitted threshold, temperature, or oracle cardinality at inference.
- Guards: finite outputs, source hashes unchanged, zero sealed accesses,
  deterministic replay, row-permutation invariance, and full tests passing.
- Forbidden changes: larger Spider, language generation, learned writers,
  sealed evaluation, or modifications to the sibling repository.

The user previously authorized unattended AutoResearch execution with no
review pauses. The evaluator is implemented as part of this milestone.

## Frozen source hashes

| File | SHA-256 |
| --- | --- |
| `train.inputs.jsonl` | `ddfd76ce63c588001d6b0f93d1a09c879d7dfdd86f9892751e6f105d9f91d207` |
| `train.labels.jsonl` | `966fe084deba95f875d0cf8a42426330df857551ac039545d19a686fab0bca21` |
| `validation.inputs.jsonl` | `3efff43b641b9e2b538c2f90947d713500f4ae5efc164f2f2d751bc418dd9cd8` |
| `validation.labels.jsonl` | `1764c9ecc44626e211af88272fe074c26defd60251253c39e7ff1d95bceb6ec1` |
| `demo.inputs.jsonl` | `5ece203ce0edcccad99dadbfe66988e05e49bd8bb6c7f8c95cd24fdb8f43a11f` |
| `demo.labels.jsonl` | `d45b4460d4ff79e00f3cc52e7da639a869cbf34f8f06ef8430126a8e7d69fce9` |

## Current approach

Each runtime case becomes one packed topology with a transient query root, 64
candidate-memory nodes, 64 directed retrieval arcs, and the observed SRE
relationships as bidirectional logical edges. Candidate enumeration uses
`PackedTopology.expand_frontier`; summary and edge content uses packed
manifold gathers; graph-neighbour aggregation uses segmented reductions.

Frozen MiniLM embeddings provide query, incoming-observation, and memory text
rows. Runtime-visible time, state, type, region, and degree features remain
separate numeric inputs. The scorer is permutation-equivariant over candidate
and manifold-row order.

## Search space

1. T0: frozen MiniLM cosine plus the registered active/supersession penalty.
2. T1: packed canonical scorer trained with balanced BCE, listwise, and
   hard-negative ranking losses, but no explicit alignment loss.
3. T2: T1 plus symmetric multi-positive query-to-memory alignment.

No other arm is permitted before these three answer the transfer question.

## History

| Iteration | Hypothesis | Status | Score | Notes |
| ---: | --- | --- | ---: | --- |
| 0 | Freeze SRE transfer protocol and source hashes. | kept | — | Sealed SRE test remains unopened. |
| 1 | T0 frozen semantic/status control. | kept as control | 0.6329 | High Recall@8, weak exact set recovery. |
| 2 | T1 packed scorer without explicit alignment. | passed | 1.0000 | Selected at step 250; exact-set accuracy 1.00. |
| 3 | T2 add multi-positive canonical alignment. | passed, not preferred | 0.9961 | Ranking passed, but exact-set accuracy fell to 0.75. |
