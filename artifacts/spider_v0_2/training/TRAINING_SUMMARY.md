# Spider v0.2 Fixed-Horizon A100 Comparison

Post-sealed architectural diagnostic only. Learned stopping is suppressed during the registered comparison, and no historical or new sealed set is opened.

| Seed | Recurrent structural | Pooled structural | R − P |
|---:|---:|---:|---:|
| 1701 | 0.3828 | 0.4609 | -0.0781 |
| 1802 | 0.3750 | 0.4453 | -0.0703 |
| 1903 | 0.3984 | 0.4531 | -0.0547 |

Mean recurrent-minus-pooled structural delta: **-0.0677**; recurrent seed wins: **0/3**.

| Model | Structural mean | Evidence exact | Evidence recall | Valid path | Mean rounds |
|---|---:|---:|---:|---:|---:|
| recurrent | 0.3854 | 0.3854 | 0.6198 | 1.0000 | 5.9688 |
| pooled | 0.4531 | 0.4531 | 0.7161 | 1.0000 | 5.9688 |

## Direct recurrent-state interventions

| Intervention | Mean degradation | Seeds ≥ 0.05 | Causal forward ablation |
|---|---:|---:|---|
| reset | +0.3854 | 3/3 | True |
| detach | +0.0000 | 0/3 | False |
| shuffle | +0.2682 | 3/3 | True |
| pooled_current_node | +0.3099 | 3/3 | False |

Recurrent-advantage rule passed: **False**.

Material-state-use rule passed: **True**.

Detach is expected to preserve forward evaluation values; it tests cross-round gradient flow only during training. Reset and graph-local shuffling are the registered causal forward tests.

All 6 runs report zero sealed access, zero deterministic replay mismatches, and zero row-permutation decision mismatches.
