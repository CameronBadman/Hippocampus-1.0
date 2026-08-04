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

## Iteration 1 — calibration-free global NULL control

- Date: 2026-08-04 Australia/Brisbane.
- Source: `d3fe4241e164154b362fad9f695c5aea05f91508`.
- Runs: `V06-Z0-s1701`, `V06-Z0-s1802`, and `V06-Z0-s1903`.
- Mean score: 0.7600 (exact 0.7809, precision 0.9322, recall 0.7600,
  scored-positive coverage 1.0000).
- Family diagnosis: corroboration recall 1.0000, latest-valid 0.9935,
  lookup 0.0000, and reachability 0.4531.
- Decision: retain as the zero-shot control. The candidate coverage ceiling is
  already one, so traversal coverage cannot explain missed lookup evidence.

## Iteration 2 — candidate-conditioned NULL

- Date: 2026-08-04 Australia/Brisbane.
- Source: `d3fe4241e164154b362fad9f695c5aea05f91508`.
- Runs: `V06-Z1-s1701`, `V06-Z1-s1802`, and `V06-Z1-s1903`.
- Mean score: 0.7608 (exact 0.7910, precision 0.9495, recall 0.7608,
  scored-positive coverage 0.9992).
- Matched outcome: one seed win; score delta +0.0008, below the registered
  0.01 minimum. Lookup recall remained 0.0000.
- Decision: reject as the finalist. Observing the current set reduces false
  positives, but plain candidate-count-weighted relative BCE does not recover
  the missed positives.

## Iteration 3 hypothesis — graph-balanced relative energy

For each graph state, average positive and negative candidate-vs-NULL losses
separately, give each present class equal mass, then average over graph states.
This tests whether large negative candidate sets dominate the boundary. It does
not alter inference: selection remains the raw comparison
`candidate_energy > null_energy` on unseen symbols.
