# Spider v0.5 Training and Reproduction

## Protocol

Spider v0.5 uses four matched arms and three seeds. Each run consumes 8,192
training cases, selects one checkpoint on 512 model-selection cases, fits its
operating policy on 512 calibration cases, and evaluates once on 1,024
development-evaluation cases. Related views remain grouped within a partition.

The maximum training length is 2,000 steps, batch size is four, and checkpoints
are written every 250 steps. The selected step is determined by exact
evidence-set accuracy subject to precision at least 0.90, then recall, macro AP,
fewer false positives, and earlier checkpoint.

## Commands

Regenerate and verify the development manifests:

```bash
.venv/bin/python scripts/generate_spider_v0_5_dataset.py
```

Run or resume one registered experiment:

```bash
.venv/bin/python scripts/run_spider_v0_5_autoresearch.py \
  --phase run --arm X0 --seed 1701
```

Generate the aggregate after all twelve records exist:

```bash
.venv/bin/python scripts/run_spider_v0_5_autoresearch.py --phase summarize
```

The orchestrator is fail-closed against concurrent campaign processes and
resumes from the latest 250-step checkpoint. A bounded subprocess timeout is
recorded as `timeout_resumable`; it is an operational event rather than an
accepted or rejected scientific result.

## Artifacts

Tracked artifacts are in `artifacts/spider_v0_5/local_rtx5070ti/`:

- `experiments.jsonl`: one complete machine-readable record per accepted run;
- `attempts.jsonl`: resumable bounded-attempt history;
- `SUMMARY.json` and `SUMMARY.md`: aggregate effects and gate results;
- `FINALIST.json`: selected arm and checkpoint identities;
- `CHECKPOINT_INDEX.json`: identities of all locally retained checkpoints; and
- `progress.png`: generated aggregate comparison plot.

Binary checkpoints and verbose per-run outputs are retained locally under
`runs/` but ignored by Git. Their SHA-256 identities, selected steps, configs,
source commit, and dataset hash are tracked. The completed local checkpoint
tree is approximately 386 MB.

## Hardware result

All registered runs completed on one NVIDIA GeForce RTX 5070 Ti in FP32. Peak
reported CUDA allocation was below 79 MB for every arm. Mean optimization
runtime ranged from approximately 124 seconds for X2 to 177 seconds for X1,
excluding checkpoint selection and evaluation.

The A100 replication gate did not open because no treatment passed the local
matched-seed advancement criteria.
