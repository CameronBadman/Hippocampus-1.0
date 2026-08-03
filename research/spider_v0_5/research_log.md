# Spider v0.5 AutoResearch log

## Iteration 0 — protocol freeze

- Timestamp: 2026-08-03 Australia/Brisbane
- Baseline commit: `92b3ea7`
- Baseline tests: 251 CPU-visible tests passed with eight CUDA skips; 259 tests
  passed with the RTX 5070 Ti visible.
- Environment: Tier 1; NVIDIA GeForce RTX 5070 Ti, driver 595.84, 16,303 MiB.
- Decision: run a full 2×2 factorial so scorer and decoder main effects remain
  identifiable. No capacity or termination change is permitted.
- Execution amendment made before results: the 250–1,000 checkpoints inside
  each full seed-1701 run are the screen. Avoiding separate 1,000-step jobs
  prevents duplicate training and preserves the registered 2,000-step maximum.

## Iteration 1 — matched control

- Arm: X0, shared pooled evidence head plus calibrated global threshold.
- Result: exact set 0.7754, precision 0.9444, recall 0.7444, coverage 0.9996.
- Family result: lookup recall 0.0000 and reachability recall 0.3672.
- Decision: keep as the matched control. The high-precision/low-recall failure
  replicated tightly across all three seeds.

## Iteration 2 — pairwise evidence matcher

- Arm: X1, pairwise query/edge/destination matcher plus global threshold.
- Result: exact set 0.7793, precision 0.9313, recall 0.7584, coverage 1.0000.
- Paired exact-set deltas: +0.0127, +0.0010, and -0.0020.
- Lookup recall remained zero on every seed.
- Decision: discard. The matcher slightly improved macro AP and mean
  reachability recall but did not meet any seed-level advancement gate.

## Iteration 3 — current-candidate count decoder

- Arm: X2, shared scorer plus current-candidate-set count selection.
- Result: raw exact set 0.8350, precision 0.7411, recall 0.9075, coverage 1.0000.
- Lookup recall rose to 0.4531 and reachability recall to 0.9557.
- False positives rose from 0.0358 to 0.2578 per case. Reachability predicted
  cardinality averaged 1.1732 against 0.5000 required.
- Decision: discard. The decoder confirms a high recall ceiling but fails the
  frozen 0.90 precision guard on all three seeds.

## Iteration 4 — matcher/decoder interaction

- Arm: X3, pairwise matcher plus current-candidate count selection.
- Result: raw exact set 0.8252, precision 0.8130, recall 0.8530, coverage 1.0000.
- Seed 1701 retained precision but lost recall; seeds 1802 and 1903 recovered
  recall but reduced precision to approximately 0.75.
- Factorial interaction on exact set: -0.0137.
- Decision: discard. The combination is unstable and does not rescue the
  precision/recall tradeoff.

## Stop decision

The registered four-iteration budget is exhausted with no advancing arm. X0
is retained. The next falsifiable hypothesis is a current-candidate,
accumulated-ledger-aware `has_evidence_now` gate followed by conditional ranked
selection. It should be tested without changing the scorer or model capacity.
No sealed data or A100 replication is permitted from this campaign.
