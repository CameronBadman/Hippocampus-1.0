# Spider v0.5 AutoResearch log

## Iteration 0 — protocol freeze

- Timestamp: 2026-08-03 Australia/Brisbane
- Baseline commit: `92b3ea7`
- Baseline tests: 251 CPU-visible tests passed with eight CUDA skips; 259 tests
  passed with the RTX 5070 Ti visible.
- Environment: Tier 1; NVIDIA GeForce RTX 5070 Ti, driver 595.84, 16,303 MiB.
- Decision: run a full 2×2 factorial so scorer and decoder main effects remain
  identifiable. No capacity or termination change is permitted.

