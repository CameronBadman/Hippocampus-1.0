# Spider v0.1 5k A100 replication

Post-sealed diagnostic only; these runs cannot change model selection, calibration, or the historical sealed result.

| Model | Seed scores | Mean | Population SD | ID evidence F1 | ID one-round stop |
|---|---|---:|---:|---:|---:|
| recurrent | 1701: 0.3622, 1802: 0.3778, 1903: 0.3778 | 0.3726 | 0.0074 | 0.4870 | 0.8698 |
| pooled | 1701: 0.4034, 1802: 0.3821, 1903: 0.3750 | 0.3868 | 0.0121 | 0.4903 | 0.8646 |

## Paired primary-metric result

| Seed | Recurrent minus pooled |
|---:|---:|
| 1701 | -0.0412 |
| 1802 | -0.0043 |
| 1903 | +0.0028 |

The mean recurrent-minus-pooled difference is `-0.0142`. The post-sealed replication therefore favors **pooled** on the registered primary metric. This is a diagnostic result, not a new selection decision.

## OOD primary autonomous success

| Split | Recurrent | Pooled | Difference |
|---|---:|---:|---:|
| validation_id | 0.3880 | 0.3672 | +0.0208 |
| validation_graph_size_ood | 0.3021 | 0.3611 | -0.0590 |
| validation_path_length_ood | 0.3507 | 0.3576 | -0.0069 |
| validation_topology_ood | 0.3715 | 0.3715 | +0.0000 |
| validation_cardinality_ood | 0.4097 | 0.4514 | -0.0417 |
| validation_equivalent_view_ood | 0.3715 | 0.3889 | -0.0174 |
| validation_composition_ood | 0.4097 | 0.4167 | -0.0069 |

All 6 accepted runs used 5,000 FP32 optimizer steps on A100 GPUs and reported zero sealed access. Every archive and standalone checkpoint was hash-verified locally and registered in [Google Drive](https://drive.google.com/drive/folders/10Pmjb0lBATNtGWyf823SB4qHAYaZ7Euw).
