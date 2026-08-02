# Spider v0.4 training protocol

All trainable arms use matched seeds 1701, 1802, and 1903, fresh row
permutations on every presentation, FP32 on the local RTX 5070 Ti, and exact
resume checkpoints. Checkpoint selection is performed at steps 250, 500, 750,
and 1,000 (and later through 2,000 where the gate calls for it).

The checkpoint-selection order is:

1. exact evidence-set accuracy subject to precision >= 0.90 and scored-positive
   coverage >= 0.98;
2. recall;
3. macro average precision;
4. fewer false positives per case;
5. better worst-positive rank;
6. earlier checkpoint.

Model-selection thresholds are temporary and are never deployed. After a
checkpoint is frozen, temperature and the final global operating threshold are
fit only on the calibration partition. The development-evaluation partition is
then evaluated once for that checkpoint.

To bound calibration compute, one full-controller reference rollout produces a
frozen-logit threshold sweep. The best few candidate thresholds are rerun
through the actual recurrent controller; only those exact reruns can become the
operating point. Thus the shortcut narrows the search but does not replace
controller feedback, evidence updates, or budget arithmetic.

Phase B holds the pooled architecture, E0 evidence objective, controller, and
512-case training protocol fixed. B0 reuses the immutable historical A0
checkpoint; B1 and B2 are the six registered new training runs.

```bash
.venv/bin/python scripts/run_spider_v0_4_autoresearch.py --phase B
```

Phase D uses the corrected, non-sealed
`spider-programs-v0.4.1-aligned-evidence-dev` dataset and holds the A2
renderer, fixed 512-case corpus, controller, E0 evidence objective, and every
model dimension constant. The five registered arms are pooled/shared (D0),
pooled/dedicated (D1), Spider/shared (D2), Spider/dedicated (D3), and
Spider/slot-aware (D4).

All 15 matched-seed arms screen under the same pre-registered 2,000-step
schedule, pausing at step 1,000. An ablation qualifies when it gains at least
0.05 exact-set accuracy or 0.07 recall, loses no more than 0.02 precision,
does not reduce macro AP, and wins at least two matched seeds. The selected
pooled and Spider arms resume exactly to step 2,000; data-order, renderer
presentation counters, optimiser state, action RNG, and PyTorch RNG are
restored from the pause checkpoint.

The slot-aware D4 evaluator demonstrated that a monolithic train, checkpoint
selection, calibration, and development-evaluation process can exceed the
five-minute process guard even though training and every checkpoint are
complete. Runs therefore use two bounded process stages: train plus checkpoint
selection, followed by resumable calibration plus development evaluation.
The pause record binds the selected checkpoint and training-source commit. This
is an orchestration boundary only; neither the model, data, thresholds, metric,
nor checkpoint-selection rule changes. The original D4 timeout remains in the
failure ledger.

```bash
.venv/bin/python scripts/run_spider_v0_4_readout.py --phase all
```

Every run records its source commit, config and dataset hashes, selected
checkpoint, calibration policy, complete per-family evidence metrics, runtime,
peak memory, and mechanical invariance guards in JSON. Crashes and timeouts are
retained separately and do not consume accepted-run budget. No v0.4 command
accepts a sealed dataset input.
