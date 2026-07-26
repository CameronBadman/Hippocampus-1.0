# Spider programs v0.2 dataset card

## Purpose

`spider-programs-v0.2` is a follow-up synthetic benchmark for autonomous
closed-loop execution. It preserves the four v0 semantics—lookup,
reachability, latest-valid, and corroboration—but uses disjoint seeds and new
manifests. It is not a natural-language or production reasoning benchmark.

## Splits

The frozen target sizes are:

| Split | Cases | Purpose |
|---|---:|---|
| train | 512 | closed-loop training |
| validation ID | 128 | threshold calibration and model selection |
| graph-size OOD | 96 | 64–128 nodes |
| path-length OOD | 96 | valid paths of length 5–8 |
| topology OOD | 96 | increased distractors and converging paths |
| manifold-cardinality OOD | 96 | threefold row-cardinality range |
| equivalent-view OOD | 96 | held-out re-keyed surface domain |
| composition OOD | 96 | multiple valid paths and longer compositions |
| rollout stress | 128 states | development-only controller states |
| test sealed | 256 | one frozen-finalist evaluation |

Every split uses `generator_version="spider-programs-v0.2"` and seeds disjoint
from v0 and from every other v0.2 split. Manifests hash specifications, case
IDs, and rollout-stress descriptors.

## Rollout-stress states

Supervisor-only stress fixtures cover:

- recoverable off-oracle frontiers;
- partial evidence ledgers;
- false-positive context reads;
- missed evidence followed by later recovery;
- premature-stop states;
- search/context/depth budget boundaries;
- duplicate and converging hypotheses.

These state descriptors are never rendered as model features.

## Leakage and invariance controls

The v0 renderer remains frozen. Row order is seeded and exchangeable, query
surfaces carry the operation, and no AST, program enum, answerability, path
membership, node ID, or trace target enters a manifold. The v0 metadata
diagnostic is rerun on new seeds and recorded with the split hash.

## Sealed policy

Spider v0 artifacts and its already-opened sealed result are immutable.
v0.1 commands reject paths under `artifacts/spider_v0` for calibration or
selection.

The v0.2 sealed cases are generated and hashed before training but are not
materialised by search commands. The primary metric, evidence threshold
procedure, source commit, finalists, and finalist seeds must be frozen before
the sole `--allow-v0-2-sealed` evaluation.
