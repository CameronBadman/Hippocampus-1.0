# Spider v0.2 Preserved-Checkpoint Diagnostic

This is a post-sealed diagnostic with no selection effect. All inputs are preserved checkpoints and non-sealed development/validation data.

## Oracle-required horizon

| Seed | Recurrent structural | Pooled structural | R − P | Recurrent final | Pooled final |
|---:|---:|---:|---:|---:|---:|
| 1701 | 0.4972 | 0.4773 | +0.0199 | 0.4119 | 0.4361 |
| 1802 | 0.4631 | 0.5284 | -0.0653 | 0.3949 | 0.4176 |
| 1903 | 0.4915 | 0.5256 | -0.0341 | 0.4205 | 0.3878 |

Mean structural delta: **-0.0265**; recurrent seed wins: **1/3**.

## Recurrent state interventions

| Intervention | Mean degradation vs intact | Seeds ≥ 0.05 |
|---|---:|---:|
| reset | +0.0089 | 0/3 |
| shuffle | +0.0000 | 0/3 |

Material state-use rule passed: **False**.

## Outcome

Recurrent-advantage rule passed: **False**.

Suppressing intermediate stopping improves final decisions for some checkpoints, but the preserved recurrent model does not show a robust structural advantage and its state interventions do not meet the pre-registered material-use threshold. Zero-shot accuracy on the new recurrence-necessity split is zero for both model families, so that split requires matched training before it can judge learnability.
