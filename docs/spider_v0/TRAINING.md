# Spider v0 training and evaluation

## Environment

The verified environment for this milestone is:

- Python 3.12;
- PyTorch 2.13.0+cu130;
- CUDA runtime 13.0;
- NVIDIA driver 595.84;
- NVIDIA GeForce RTX 5070 Ti;
- CUDA BF16 support available.

`uv sync --extra test` creates/synchronises the project environment. Explicit
CUDA BF16 packing is used only after hardware validation; there is no FP16
fallback.

## Reproducible commands

```bash
uv sync --extra test
.venv/bin/pytest -q
.venv/bin/python scripts/generate_spider_dataset.py
```

Tiny overfit:

```bash
.venv/bin/python scripts/train_spider_v0.py \
  --config configs/spider_v0/tiny_overfit.json \
  --output-dir artifacts/spider_v0/runs/tiny_overfit_cuda \
  --device cuda --dtype float32
```

BF16 CUDA smoke:

```bash
.venv/bin/python scripts/train_spider_v0.py \
  --config configs/spider_v0/tiny_overfit.json \
  --output-dir artifacts/spider_v0/runs/bf16_cuda_smoke \
  --device cuda --dtype bfloat16 --steps 40 --cases 16
```

AutoResearch:

```bash
.venv/bin/python scripts/run_spider_autoresearch.py \
  --steps 60 --train-cases 48 --eval-cases 8 \
  --device cuda --dtype float32
```

Validation evaluation:

```bash
.venv/bin/python scripts/evaluate_spider_v0.py \
  --config artifacts/spider_v0/autoresearch/configs/E003-recurrent-standard.json \
  --checkpoint artifacts/spider_v0/autoresearch/runs/E003-recurrent-standard/checkpoint.pt \
  --split validation_path_length_ood --cases 32
```

The sealed command requires `--allow-sealed` and was invoked exactly once for
the frozen finalist.

## Stages

### Stage A: fixed tiny overfit

The current 48-case fixture covers all four program families and all six
termination outcomes. It uses oracle frontiers and trains every candidate and
termination head. The final CUDA FP32 run reached:

- candidate Top-1 and MRR: 1.000;
- expand/context/evidence/support/conflict accuracy: 1.000;
- termination accuracy: 1.000;
- invalid expansion rate: 0.000.

The checkpoint itself is ignored by Git; its SHA-256 manifest and complete
history/metrics are tracked in
`artifacts/spider_v0/runs/tiny_overfit_cuda_v3/`.

### Stage B: oracle-frontier structural training

`oracle_rollout` expands the current packed hypothesis frontier with
`PackedTopology.expand_frontier`, maps every packed candidate back to its
set-valued oracle target, gathers packed manifolds, refines positive
context-read candidates, and carries selected path manifolds into the next
round. Multiple acceptable expansions are trained with a segmented
multi-positive listwise loss.

### Stage C: scheduled model rollouts

`mixed_rollout` chooses oracle or deterministic model frontiers from a seeded
fraction. Off-oracle candidates receive explicit hard-negative targets.
`TrainingLoopConfig.oracle_fraction_schedule` supports the configured
`[1.0, 0.75, 0.5, 0.25]` progression without differentiating through top-k.

The short AutoResearch matrix was run before this final scheduled-rollout
correctness extension, so its evidence primarily measures Stage B. This is a
material limitation and the first recommended follow-up.

## Loss accounting

Every loss records raw value, weighted value, and target count:

- multi-positive frontier priority;
- expand/retain;
- context value-of-information;
- evidence inclusion;
- support/conflict;
- remaining cost;
- explicit termination reason;
- optional behavioural consistency;
- optional search/context cost.

Candidate losses use FP32 numerics for FP16/BF16 model outputs. The trainer logs
the clipped total gradient norm. Zero-opportunity losses return differentiable
zero.

## Attention backend

The correctness backend is padded, position-free PyTorch SDPA. It uses no
positional encoding and no causal mask, and it bypasses all-empty memories.

PyTorch 2.13 exposes ordinary public SDPA but no stable public packed-varlen
attention contract used by this project. The private variadic symbol is not
treated as a supported API, so Spider reports CUDA varlen execution as
unavailable and keeps the verified padded path. The data layer's
`VarlenManifoldBatch` remains available for a future supported consumer.

## Checkpoint and sealed-test policy

Binary checkpoints are local/ignored because they are reproducible and much
larger than their manifests. Every retained run records:

- config and source commit;
- seed and split digest;
- parameter count and steps;
- device/runtime;
- checkpoint SHA-256;
- full metrics and failure reason.

`artifacts/spider_v0/selected_checkpoint.json` identifies the frozen finalist
and records the one sealed report digest.
