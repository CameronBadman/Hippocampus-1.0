# Spider v0.6 Training and Reproduction

## Protocol

Four arms were trained with matched seeds 1701, 1802, and 1903. Every run used
8,192 training cases, 512 model-selection cases, a diagnostic-only 512-case
calibration partition, and one evaluation over 1,024 development cases.
Training used FP32 on one NVIDIA GeForce RTX 5070 Ti, batch size four, at most
2,000 steps, and checkpoints every 250 steps.

Checkpoint selection used the frozen weakest-metric objective and registered
tie-breakers. It did not fit a temperature, threshold, or cardinality policy.
The inference rule was always `candidate_energy > null_energy`.

## Commands

Regenerate and validate the symbol-disjoint manifests:

```bash
.venv/bin/python scripts/generate_spider_v0_6_dataset.py
```

Run or resume one registered arm and seed:

```bash
.venv/bin/python scripts/run_spider_v0_6_autoresearch.py \
  --phase run --arm Z2 --seed 1701
```

Generate the aggregate after all twelve run records are present:

```bash
.venv/bin/python scripts/run_spider_v0_6_autoresearch.py --phase summarize
```

The orchestrator holds a campaign lock and resumes training, selection, or
evaluation from durable stage markers. Bounded timeouts are recorded as
resumable operational attempts rather than scientific failures.

## Runtime and artifacts

Mean optimization runtimes for Z0 through Z3 were approximately 118, 121, 131,
and 136 seconds respectively. Peak reported CUDA allocation was below 75 MiB
for every arm. Selection and evaluation run after optimization and are not
included in those means.

Tracked outputs in `artifacts/spider_v0_6/local_rtx5070ti/` include:

- `experiments.jsonl`: twelve machine-readable accepted run records;
- `attempts.jsonl`: bounded resumable-attempt history;
- `SUMMARY.json` and `SUMMARY.md`: aggregate and per-seed gates;
- `FINALIST.json`: the accepted-control checkpoint identities;
- `BEST_OBSERVED.json`: the non-accepted Z2 diagnostic checkpoint identities;
- `progress.png`: generated score history; and
- checkpoint SHA-256 values, source commits, dataset hash, runtime, and memory
  in the ledger.

Binary checkpoints and verbose per-run files remain local under `runs/` and
are ignored by Git. No A100 replication ran because no candidate passed the
local gate.
