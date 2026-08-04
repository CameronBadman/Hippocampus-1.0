# Spider v0.7 dataset card

## Identity and purpose

- Version: `spider-programs-v0.7-binding-dev`
- Aggregate SHA-256: `240d5c0e222f1ca6f056bf0b31140a094d126e7d2695916cdfc1d8815a199409`
- Purpose: development-only zero-shot relation/value binding and exact
  evidence selection.
- Renderer: `renderer-v0.4`, orthogonal-aligned geometry.
- Sealed cases: none materialised or evaluated.

## Partitions

| Partition | Cases | Observable symbols | Matched lookup pairs |
| --- | ---: | ---: | ---: |
| Training | 8,192 | 595,612 | 1,024 |
| Model selection | 512 | 37,227 | 64 |
| Calibration diagnostic | 512 | 37,233 | 64 |
| Development evaluation | 1,024 | 74,445 | 128 |

All 744,517 observable symbols are partition-local; measured cross-partition
overlap is zero. The calibration-named partition is retained for structural
compatibility with the existing training harness but v0.7 does not fit a
temperature, threshold, count, or family-specific policy from it.

## Lookup correction

Each lookup state contains four structurally matched candidates: correct
relation/value, correct relation with a wrong value, wrong relation with the
correct value, and correct relation/value behind an invalid gate. Every edge
has the same row count and every destination summary is padded to the same row
count.

Answerable and absent cases are generated in matched pairs. A pair has equal
query observations, topology, graph size, node and edge cardinalities, scalar
inventory, and overall symbol inventory. The absent member changes the former
positive candidate to a wrong-relation/wrong-value binding while retaining all
affected symbols elsewhere.

## Integrity checks

Generation rejects duplicate case/base identities, cross-partition symbol
reuse, supervisor or evidence-label mismatches, unsupported cases, query-row
answerability leakage, malformed lookup pairs, and any sealed specification.
The manifests hash the complete generated case content rather than only seeds
or configuration.

This synthetic benchmark measures controlled graph binding. It is not a
natural-language, real-retrieval, or production validation dataset.
