# Spider v0.6 AutoResearch log

## Iteration 0 — protocol freeze

- Date: 2026-08-04 Australia/Brisbane.
- Baseline source: `3bcf9c3`.
- Environment: Tier 1; NVIDIA GeForce RTX 5070 Ti, driver 595.84,
  16,303 MiB.
- Prior diagnosis: scored-positive coverage is effectively one; calibrated
  thresholding misses positives, while count selection recovers them by
  purchasing false positives.
- Decision: make NULL a candidate-relative energy and remove all held-out
  calibration from inference. Preserve model size and traversal so the causal
  comparison isolates the decision rule.

