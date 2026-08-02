# Spider v0.4 Phase E: Frozen-Logit Set-Decoding Ceiling

## Decision

Advance the set-decoding branch (Phase F1) and do not run the ranking-loss
branch first.

This diagnostic reuses the immutable candidate observations from the selected
Phase D development evaluations. It does not rerun the model, recalibrate on
development evaluation, or access a sealed split.

## Results

| Frozen arm | Global threshold exact set | Oracle-cardinality top-k | Gain |
| --- | ---: | ---: | ---: |
| D0 pooled/shared | 0.7549 | 0.9274 | +0.1725 |
| D2 Spider/shared | 0.7305 | 0.8919 | +0.1615 |

For the primary D0 arm, the three seed-level gains were `+0.1865`, `+0.1777`,
and `+0.1533`. All exceed the preregistered `+0.15` branch gate. The result
means the frozen scorer usually ranks the required set well enough for a much
better decoder, while the calibrated global threshold loses substantial exact
set accuracy.

The per-case optimal-threshold ceiling equalled the oracle-cardinality ceiling
in aggregate. The oracle null/no-evidence diagnostic produced only a small
change from the global threshold, so detecting empty evidence sets alone is not
the dominant error.

## Integrity

- Dataset: `spider-programs-v0.4.1-aligned-evidence-dev`
- Dataset hash: `8ff3c7f12978e8381552eafadbe5fc6dfab8eb08c2484204e1cdad7835dc8a32`
- Primary arm: D0 pooled/shared
- Development-evaluation reexecution: no
- Sealed accesses: zero
- Next registered arms: global threshold, learned null, learned cardinality,
  and learned null plus cardinality

Machine-readable seed reports and the branch decision are under
`artifacts/spider_v0_4/phase_e/frozen_policies/`.
