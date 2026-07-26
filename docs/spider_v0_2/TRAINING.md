# Spider v0.2 Training Protocol

## Primary matched comparison

Train the existing recurrent-standard Spider and pooled control on
`train_recurrence_necessity` with identical renderer, controller, optimiser,
case order, action schedule, and loss weights.

- Train cases: all 512 development cases.
- Validation cases: all 128 held-out development cases.
- Seeds: 1701, 1802, 1903.
- Optimizer steps: 6,000.
- Cases per optimizer step: 4.
- Effective training examples: 24,000 per run.
- Recurrent horizons: oracle-required, four to eight rounds.
- Precision: FP32.
- Accelerator: A100 or H100.
- Checkpoint interval: 1,000 optimizer steps.

The fixed horizon affects execution only. Candidate controls retain the same
configured round normalisation, and the horizon is never passed as a neural
feature.

## Loss and action schedule

Termination loss has zero weight in this architectural comparison. The
terminator is evaluated at the final horizon for diagnostics but cannot
suppress state transitions.

Evidence uses class-balanced candidate BCE plus the existing set-level
positive-mass/soft-recall objective. Search-cost and context-cost penalties are
zero. Context is absent from this dataset.

Frontier, context, and evidence action teacher forcing moves through:

1. 1.00;
2. 0.75;
3. 0.50;
4. 0.25;
5. 0.00.

Termination action sourcing remains model-only but is ignored before the final
horizon. Every encountered candidate state receives supervision.

## Evaluation

Calibrate the evidence threshold on
`validation_recurrence_necessity` only. This split is development data and is
not sealed.

Report:

- fixed-horizon structural success;
- exact evidence-edge set accuracy;
- evidence precision, recall, F1, and average precision;
- valid-path and trace validity;
- final autonomous success after the horizon;
- arcs scored and rounds;
- deterministic replay and row-permutation mismatches.

For recurrent runs also evaluate the preserved path-state interventions:
`none`, `reset`, `detach`, `shuffle`, and `pooled_current_node`.

## Checkpoint durability

Each run writes `checkpoint_step_001000.pt` through
`checkpoint_step_005000.pt` and a final `checkpoint.pt`. A remote run is not
released until:

1. its archive and standalone final checkpoint are downloaded;
2. local SHA-256 verification passes;
3. both are uploaded to the registered Google Drive experiment folder;
4. Drive metadata and byte counts are verified;
5. the Drive file IDs are committed to the experiment ledger.

Intermediate checkpoints remain inside the verified run archive. This keeps
the Drive folder manageable while preserving recovery points.

## Commands

Smoke:

```bash
.venv/bin/python scripts/train_spider_recurrence.py \
  --config configs/spider_v0_2/recurrent_recurrence.json \
  --experiment-id smoke-recurrent \
  --output-dir /tmp/spider-v02-smoke \
  --steps 2 --train-cases 4 --eval-cases 4
```

Full recurrent seed:

```bash
.venv/bin/python scripts/train_spider_recurrence.py \
  --config configs/spider_v0_2/recurrent_recurrence.json \
  --experiment-id R-REC-s1701-6k \
  --output-dir artifacts/spider_v0_2/training/R-REC-s1701-6k \
  --seed 1701
```

