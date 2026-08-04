# Spider v0.8 final report: SRE graph-retrieval transfer

## Result

The packed canonical retriever solved this synthetic development transfer, but
the explicit alignment-loss hypothesis did not win. T1, which omits the extra
alignment loss, achieved a 1.000 registered ranking score on all three seeds.
It averaged 0.983 exact evidence-set accuracy, 0.991 precision, and 1.000
recall. The frozen semantic/status control scored 0.633. T2 retained excellent
ranking but reduced exact-set accuracy to 0.857.

| Arm | Seeds | Primary score | Exact set | Precision | Recall | FP/case |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T0 frozen control | 1 | 0.6329 | 0.0000 | 0.0743 | 0.9675 | 29.670 |
| T1 packed scorer | 3 | **1.0000** | **0.9833** | **0.9907** | **1.0000** | **0.023** |
| T2 + alignment loss | 3 | 0.9982 | 0.8567 | 0.9257 | 0.9837 | 0.200 |

T1's four registered ranking components—MRR, Recall@8, macro AP, and macro
hard-negative pairwise accuracy—were each 1.000 on every seed. Clock skew,
regional failover, and secret rotation each had 1.000 MRR, Recall@8, and AP.
The selected steps were 250, 500, and 500; later checkpoints were not assumed
to be better.

## What the ablations show

Frozen-checkpoint ablations were run only after the finalist was selected.
They were diagnostic and did not change the selection.

| Input intervention | Primary score | Exact set | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Full T1 | 1.0000 | 0.9833 | 0.9907 | 1.0000 |
| Remove runtime features | 0.9595 | 0.5467 | 0.8373 | 0.8523 |
| Remove semantic embeddings | 0.6676 | 0.0000 | 0.3092 | 0.8320 |
| Shuffle candidate semantics | 0.7457 | 0.2900 | 0.5240 | 0.6382 |
| Shuffle runtime features | 0.9019 | 0.5100 | 0.6794 | 0.7751 |
| Supply another case's query | 0.5940 | 0.2200 | 0.3322 | 0.3455 |

This falsifies the simplest metadata-shortcut explanation. Query-conditioned
semantics are necessary, while runtime time/status/type and graph controls are
also needed for precise set boundaries. Semantics alone rank well but do not
recover the exact evidence set reliably.

It does **not** establish that the explicit T2 loss transfers: T1 consistently
outperformed it. The shared canonical projection can still align under the
ordinary retrieval objectives, so the negative finding is specifically about
the additional multi-positive alignment objective in this configuration.

## Execution and reproducibility

Every learned score was produced after `PackedTopology.expand_frontier`
enumerated the 64 root-to-memory arcs. Candidate and neighbor values came from
packed manifold gathers, and graph-neighbor state used segmented reductions.
No second Python graph participated in inference.

All seven registered runs were mechanically accepted. The three T1 runs had:

- zero bitwise score, NULL-score, or decision mismatches on repeated
  deterministic evaluation;
- zero candidate-order decision mismatches, with maximum floating score delta
  `1.91e-6`;
- 100% enumerated-candidate and scored-positive coverage;
- maximum peak allocated CUDA memory of 173.4 MiB;
- mean deterministic evaluation latency of 6.46 ms per case;
- zero sealed accesses.

The three public demo cases also recovered their published evidence sets
exactly (4/4, 1/1, and 2/2) without contributing to model selection or an
aggregate metric.

The selected model-only weights are committed as a hash-checked safetensors
artifact; optimizer state and embedding caches remain local build products.

## Research conclusion

The actionable result is a small, calibration-free SRE evidence retriever that
combines frozen natural-language semantics with observable graph and validity
controls. It provides a credible one-day demo path and is stronger than the
frozen pooled control on this dataset.

The result is not production validation. All language and incident worlds are
synthetic, the development evaluation contains only 100 cases, and its unseen
scenario families still share the upstream generator and renderer. No SRE
sealed split was opened, and no A100 replication was run because this campaign
was explicitly kept local. Perfect development ranking also suggests the
benchmark is now near its ceiling.

The next useful experiment is not a larger Spider. Freeze T1 and evaluate it
on an independently authored or carefully redacted incident-memory set with
different templates, then add counterfactual pairs that preserve status/type
statistics while changing the query-dependent evidence relation. If that
transfer holds, integrate the retriever into the SRE/Entor demo with citations
to the exact evidence ledger. If it fails, improve data diversity and
supervision before adding model capacity.
