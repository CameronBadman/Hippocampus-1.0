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

## Iteration 3 result — graph-balanced relative energy

- Date: 2026-08-04 Australia/Brisbane.
- Source: `6802f87ff6d784e3b28b8d3be1469060b33c4501`.
- Runs: `V06-Z2-s1701`, `V06-Z2-s1802`, and `V06-Z2-s1903`.
- Mean score: 0.7998 (exact 0.7998, precision 0.8722, recall 0.8021,
  scored-positive coverage 1.0000).
- Matched outcome: two seed wins over Z0. Reachability recall increased from
  0.4531 to 0.7161, while lookup recall remained 0.0026.
- Decision: keep as the parent for iteration 4 because the score improved by
  0.0398. Do not select it as a finalist: it misses the 0.82 target, and
  aggregate precision also falls below the inherited 0.90 diagnostic floor.

## Iteration 4 hypothesis — bounded hard-negative NULL margin

Retain graph-balanced BCE and add a modest loss that places required positives
above NULL and only the highest-scoring structurally plausible negatives below
NULL. Bound the negative set per graph so easy-negative count cannot dominate.
This is the final registered iteration and does not change inference.

## Iteration 4 result — bounded hard-negative NULL margin

- Date: 2026-08-04 Australia/Brisbane.
- Source: `992d98a8f2c057713041faf394d31f9355bde0e1`.
- Runs: `V06-Z3-s1701`, `V06-Z3-s1802`, and `V06-Z3-s1903`.
- Mean score: 0.7937 (exact 0.7982, precision 0.8906, recall 0.7937,
  scored-positive coverage 0.9988, macro AP 0.9269).
- Lookup recall increased from 0.0026 in Z2 to 0.0833, but lookup precision
  was 0.2874 and reachability recall fell from 0.7161 to 0.5833.
- Decision: reject. Z3 regressed 0.0061 from Z2, did not pass the target, and
  remained below 0.90 aggregate precision. The iteration budget is exhausted.

## Campaign conclusion

The best measured zero-shot selector was Z2 at score 0.7998, but it is not an
accepted finalist because precision was 0.8722 and the target was 0.82. Z0
remains the accepted control. Candidate coverage was effectively complete and
all arms retained deterministic replay and row-permutation invariance. The
remaining lookup failure is a scorer/representation-readout problem rather
than candidate enumeration, and further NULL-loss tuning is not justified by
this campaign.
