# Spider v0.2 Continuation Handoff

## Resume point

Repository `main` contains all implementation, tests, the complete accepted
three-seed recurrent/pooled matrix, generated aggregate reports and plots, and
42 verified Drive records. No further registered training run remains.

The cancelled 3,000-step seed-1802 attempt remains preserved as excluded audit
evidence. Its replacement restarted from step zero and completed all 6,000
steps as `REC-recurrent-s1802-6k`.

The fail-closed aggregator accepted all six runs. Do not tune this completed
matrix, rerun selection, change the fixed horizon, or read historical sealed
artifacts.

## Frozen identities

- Source/model commit:
  `acb533666d481daf9b6fb56562d69a5dd78c5e0e`
- Reusable worker commit:
  `c4814ed7ac3073724b8cc23d33b25dacae6c46a6`
- Worker SHA-256:
  `0bb62d1491f75bf604c0c8353dd93472dcc5cd91df10e9e151e4d929094ed9c8`
- Training split SHA-256:
  `ff36529a8090581f6156a8fc36258e4a14eee9a542955623b70550001469fe56`
- Validation split SHA-256:
  `67c2273e4899af179bc1e10185742b806d751f5f5dba858c771f2eca8a6af4aa`

Launch-source hashes:

| Spec | SHA-256 |
| --- | --- |
| `recurrent_1802.py` | `cc6bd0451c6fbefef849459a1933d3651ef071c06efaa03e923c4fe43c687774` |
| `pooled_1802.py` | `410d27f2a2554bfd249409afba7ae2610567f73b00b33e3fe3193a203ae4b983` |
| `recurrent_1903.py` | `7ad2b6dccde3f16ea0ce05e58d2ca377aaeeb7ddad3386e1e0d3509d204f9b58` |
| `pooled_1903.py` | `3b6e5d46a9ab27d2d5809c3429d495e98203ac3c1240991151defde6b95d576e` |

## Completed verification

From the repository root, reproduce the accepted aggregate with:

```bash
git switch main
git pull --ff-only
.venv/bin/pytest -q
.venv/bin/python scripts/aggregate_spider_v0_2_training.py
.venv/bin/python scripts/render_spider_v0_2_training_plots.py
```

The aggregator must fail closed unless source/data identities match, every
checkpoint is present, sealed access is zero, and deterministic replay and row
permutation mismatches are zero. The certified conclusion is in
`docs/spider_v0_2/FINAL_REPORT.md`.

Better Colab reports the completed pooled endpoint as backend-dead and
kernel-disconnected. Google had already unassigned it before the explicit stop,
so both stop commands returned `Not Found`; the remaining CLI entry is stale
local bookkeeping and must not be confused with a live accelerator.
