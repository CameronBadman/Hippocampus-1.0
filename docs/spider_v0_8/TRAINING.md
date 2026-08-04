# Spider v0.8 training and reproduction

Spider v0.8 uses one local NVIDIA GeForce RTX 5070 Ti in FP32. The frozen
MiniLM encoder runs only while preparing the cache; its parameters are never
trained. Each learned run contains 1,000 optimization steps, batch size 32,
and model-selection checks at steps 250, 500, 750, and 1,000. Selection uses
only the 100-case model-selection partition. The 100-case development-
evaluation partition is opened once for each selected checkpoint.

Install the test and SRE extras, then build the validated embedding cache:

```bash
uv sync --extra test --extra sre
.venv/bin/python scripts/prepare_spider_v0_8_sre.py --device cuda
```

Run the registered screen and confirmation matrix:

```bash
.venv/bin/python scripts/run_spider_v0_8_autoresearch.py --stage screen
.venv/bin/python scripts/run_spider_v0_8_autoresearch.py --stage confirm
```

The first command evaluates T0 and trains T1/T2 with seed 1701. The second
trains T1/T2 with seeds 1802 and 1903. Every trainable run is independently
bounded to 295 seconds and stores all checkpoint-selection observations.
Crashes and guard failures remain in the JSONL ledger and do not count as
accepted results.

Re-evaluate the frozen finalist and run its public terminal demo with:

```bash
.venv/bin/python scripts/evaluate_spider_v0_8_sre.py
.venv/bin/python scripts/demo_spider_v0_8_sre.py
```

The evaluator allows only `selection` or `evaluation`; it has no sealed-test
option. The shared loader rejects paths named `test.*` or contained in sealed
directories. The demo uses three independent public cases and deliberately
does not emit an aggregate research metric.

## Mechanical guards

An accepted run requires finite metrics, complete candidate enumeration,
complete scored-positive coverage, zero sealed accesses, zero exact replay
mismatches, and zero row-permutation decision mismatches. Deterministic
evaluation explicitly enables PyTorch deterministic algorithms and the packed
segmented deterministic path; training keeps the faster registered execution
mode.

Artifacts are recorded in
[`experiments.jsonl`](../../artifacts/spider_v0_8/local_rtx5070ti/experiments.jsonl),
the generated [experiment summary](../../artifacts/spider_v0_8/local_rtx5070ti/EXPERIMENTS.md),
and the [selected checkpoint manifest](../../artifacts/spider_v0_8/SELECTED_CHECKPOINT.json).
Large embedding caches and optimizer checkpoints are intentionally gitignored.
The selected model-only safetensors file is versioned and hash-checked so the
demo does not depend on an ignored training-run directory.
