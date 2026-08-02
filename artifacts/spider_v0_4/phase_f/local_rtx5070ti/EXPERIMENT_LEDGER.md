# Spider v0.4 Phase F1 experiment ledger

| Arm | Seed | Status | Step | Exact set | Precision | Recall | Coverage | Cardinality MAE | Source |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| F0 | 1701 | reused | 2000 | 0.7480 | 0.8971 | 0.7440 | 0.9976 | 0.2637 | `ab28a708` |
| F0 | 1802 | reused | 1500 | 0.7549 | 0.8552 | 0.7596 | 1.0000 | 0.2842 | `ab28a708` |
| F0 | 1903 | reused | 750 | 0.7617 | 0.8967 | 0.7512 | 1.0000 | 0.2568 | `ab28a708` |
| F1 | 1701 | accepted | 500 | 0.7598 | 0.9787 | 0.7163 | 0.9820 | 0.2412 | `41d78cad` |
| F1 | 1802 | accepted | 1000 | 0.7666 | 0.7381 | 0.8197 | 1.0000 | 0.3770 | `0b740e3f` |
| F1 | 1903 | accepted | 1250 | 0.7637 | 0.8895 | 0.7740 | 0.9988 | 0.2461 | `0b740e3f` |
| F2 | 1701 | accepted | 1000 | 0.4932 | 0.6206 | 0.4267 | 1.0000 | 0.4609 | `0b740e3f` |
| F2 | 1802 | accepted | 1250 | 0.4668 | 0.5654 | 0.5505 | 0.9988 | 0.4863 | `0b740e3f` |
| F2 | 1903 | accepted | 1000 | 0.4814 | 0.5846 | 0.5024 | 1.0000 | 0.4326 | `0b740e3f` |
| F3 | 1701 | accepted | 1500 | 0.6094 | 0.8298 | 0.4748 | 0.9988 | 0.3730 | `0b740e3f` |
| F3 | 1802 | accepted | 750 | 0.5117 | 0.9350 | 0.3113 | 1.0000 | 0.5420 | `0b740e3f` |
| F3 | 1903 | accepted | 1750 | 0.5850 | 0.8670 | 0.4543 | 1.0000 | 0.4219 | `0b740e3f` |

One F1/1802 attempt was invalidated after concurrent execution was detected. Both processes were terminated, the partial checkpoints were quarantined, and the seed was rerun from step zero under the campaign lock. The invalid attempt is retained in `failed_experiments.jsonl` and is not counted above.

No sealed split was accessed.
