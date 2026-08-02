# Spider v0.4 Frozen Design

## Scope

Spider v0.4 isolates representation geometry, data reuse, evidence readout,
and set decoding. It does not scale the graph processor or reopen termination.
Historical v0-v0.3 artifacts and every sealed split are immutable.

## Renderer geometries

`renderer-v0.4` exposes three deterministic geometries over an opaque symbol
latent `z`:

- A0 samples independently from `(modality, symbol)` and is the historical
  negative control.
- A1 emits `z(symbol) + e(modality)` and is an intentionally easy upper bound.
- A2 emits `R(modality) z(symbol) + e(modality)`, where every `R` is a fixed,
  seeded orthogonal matrix.

Scalars are first represented in the same shared latent space and then follow
the same modality transform. No answer, path, node, relation, program-family,
or row-position feature is introduced. A2 must generalise through a small
probe to symbols absent from probe training before any Spider training starts.

## Data boundary

The development dataset has 8,192 training, 512 model-selection, 512
calibration, and 1,024 development-evaluation base cases. Partitioning is by
base-case identity. Family, answerability, required-evidence cardinality, path
length, and graph-size strata are measured and balanced where the generator's
family semantics permit.

Every training presentation receives a fresh deterministic row permutation.
The online arm additionally receives fresh cases and symbols. Exact resume
must restore presentation counters and generated-case position.

## Readout boundary

The shared seven-output head is the historical control. A dedicated pooled
evidence head gets its own LayerNorm and MLP. The slot-aware head uses one
learned evidence query to cross-attend, without positions, over path, query,
source, edge, destination, global-evidence, and controller rows. It changes
only the evidence logit.

## Evaluation boundary

Checkpoint selection uses only model-selection data. Temperature and the
operating policy use only calibration. Development evaluation is opened once
per selected checkpoint. Frozen-logit oracle policies diagnose ranking and
cardinality ceilings but are never deployable results.

Only one post-diagnostic branch runs: learned null/cardinality decoding when
oracle-k improves exact-set accuracy by at least 0.15, otherwise clean ranking
with mechanically guaranteed hard negatives. G2/G3 are invalid when their
plausible-negative target count is zero.
