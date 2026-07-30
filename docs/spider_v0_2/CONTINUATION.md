# Spider v0.2 Continuation Handoff

## Resume point

Repository `main` contains all implementation, tests, the accepted seed-1701
pair, the accepted recurrent seed-1802 restart, plots tooling, and verified
Drive records. The recurrent seed-1802 A100 session was released after its
artifacts were verified; it is no longer consuming Colab credits.

The cancelled 3,000-step seed-1802 attempt remains preserved as excluded audit
evidence. Its replacement restarted from step zero and completed all 6,000
steps as `REC-recurrent-s1802-6k`.

Remaining accepted-run order:

1. pooled seed 1802;
2. recurrent seed 1903;
3. pooled seed 1903.

Do not overlap these runs, tune between them, change the fixed horizon, or read
historical sealed artifacts.

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

## Launch pooled seed 1802

From the repository root:

```bash
git switch main
git pull --ff-only
.venv/bin/pytest -q

sha256sum scripts/colab_spider_v0_2_specs/pooled_1802.py
better-colab session ensure spider-v02-pool-p1802 --gpu A100 --format json
better-colab execution start \
  --session spider-v02-pool-p1802 \
  --file scripts/colab_spider_v0_2_specs/pooled_1802.py \
  --expected-source-sha256 \
    sha256:410d27f2a2554bfd249409afba7ae2610567f73b00b33e3fe3193a203ae4b983 \
  --idempotency-key spider-v02-pool-p1802-6k-v1 \
  --execution-timeout 18000 \
  --detach \
  --format json
```

Use H100 instead of A100 only if it is explicitly allocated and visible; the
worker accepts either and rejects lower accelerators.

Record the returned execution ID, then monitor without rerunning the source:

```bash
better-colab execution wait EXECUTION_ID --timeout 50
better-colab execution output EXECUTION_ID --format json
```

When paging output, pass the last returned cursor with `--cursor`. Download
each stable `display_data` checkpoint, verify its published SHA-256 and
checkpoint payload, upload it to the registered Drive folder, verify Drive
size/parent metadata, add it to
`artifacts/spider_v0_2/GOOGLE_DRIVE_BACKUP.json`, and commit/push a meaningful
checkpoint record.

## Launch the other remaining runs

Repeat the same command shape using the exact spec/hash pairs above and fresh
idempotency keys:

```text
spider-v02-rec-r1903-6k-v1
spider-v02-pool-p1903-6k-v1
```

Do not launch the next run until the preceding result ZIP and standalone final
checkpoint have been downloaded, deeply verified, uploaded to Drive, recorded,
committed, and pushed. Stop each session after its artifacts are safe.

## Accept and report the complete matrix

For every extracted result:

```bash
.venv/bin/python scripts/verify_spider_v0_2_recurrence_run.py \
  artifacts/spider_v0_2/training/isolated/MODEL_SEED/EXPERIMENT_ID
```

After all six exact runs and all 42 Drive artifacts exist:

```bash
.venv/bin/python scripts/aggregate_spider_v0_2_training.py
.venv/bin/python scripts/render_spider_v0_2_training_plots.py
.venv/bin/pytest -q
git status --short
```

The aggregator must fail closed unless source/data identities match, every
checkpoint is present, sealed access is zero, and deterministic replay and row
permutation mismatches are zero. Only then replace the interim report with the
three-seed conclusion in `docs/spider_v0_2/FINAL_REPORT.md`, commit, and push.
