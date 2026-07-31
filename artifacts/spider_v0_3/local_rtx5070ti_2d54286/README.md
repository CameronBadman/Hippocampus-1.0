# Spider v0.3 local CUDA contingency

This directory preserves the complete aggregate evidence from source
`2d542860af20bfe6ef2ab64e2950df1d07ceb2da` on an NVIDIA GeForce RTX
5070 Ti. It is not labelled as the preregistered A100 result.

- `evidence/` contains 12 accepted-run ledgers, decisions, logs, calibration
  records, and detailed per-run metrics.
- `termination/` contains nine accepted-run ledgers, decisions, logs, and
  detailed per-run metrics.
- `checkpoints/` contains all 21 final checkpoints locally. Git ignores those
  binary files; they are backed up to the Drive folder recorded in
  `GOOGLE_DRIVE_CHECKPOINTS.json`.

Every aggregate records zero sealed-set access. The final interpretation is in
`docs/spider_v0_3/FINAL_REPORT.md`.
