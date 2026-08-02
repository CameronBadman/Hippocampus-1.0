# Spider v0.4 Phase F1 Preregistration: Evidence Set Decoding

## Frozen causal decision

Phase E measured a `+0.1725` mean oracle-cardinality exact-set ceiling for the
primary pooled D0 arm, with every matched seed exceeding the registered
`+0.15` branch gate. Phase F therefore runs learned set decoding before any
ranking-loss or hard-negative experiment. The Phase E observations and branch
decision are immutable.

## Fixed inputs

- Dataset: `spider-programs-v0.4.1-aligned-evidence-dev`
- Dataset hash: `8ff3c7f12978e8381552eafadbe5fc6dfab8eb08c2484204e1cdad7835dc8a32`
- Renderer: A2 orthogonal-aligned, seed 91337
- Model: D0 pooled processor with shared evidence scorer
- Training corpus: fixed 512-case protocol with fresh row permutation per
  presentation
- Steps: at most 2,000; checkpoints every 250 steps
- Seeds: 1701, 1802, 1903
- Execution: fixed oracle-required horizon; learned frontier and evidence
  actions; learned stopping remains disabled
- Sealed access: forbidden

F0 reuses the compatible immutable Phase D D0 full-run checkpoint selections
and development observations. It is not retrained or evaluated again.

## Arms

| Arm | Evidence selection | Added prediction | Added loss |
| --- | --- | --- | ---: |
| F0 | calibrated global threshold | none | 0 |
| F1 | candidate logit above graph null | learned null boundary | 1.0 |
| F2 | total cardinality minus already-recorded evidence, then stable top-k | 0/1/2/3/4+ cardinality | 1.0 |
| F3 | null filtering followed by remaining-cardinality top-k | both heads | 0.5 each |

The F3 weights hold the aggregate auxiliary-loss budget approximately constant
rather than silently doubling it. All policies retain the exact evidence
budget and stable snapshot-local tie-breaking. Predicted cardinality is total
required cardinality; evidence already present in the exact ledger is
subtracted at every controller round.

## Selection and gate

Checkpoint selection uses only model selection, in this order:

1. exact set accuracy with precision >= 0.90 and scored-positive coverage >=
   0.98;
2. evidence recall;
3. macro average precision;
4. fewer false positives per case;
5. better worst-positive rank;
6. earlier checkpoint.

For learned policies, a threshold sweep has no operational meaning. The
calibration split therefore performs one exact controller evaluation at the
registered policy and records identity temperature/threshold placeholders.
Development evaluation is opened once for the selected checkpoint.

An arm advances only when at least two of three matched seeds each achieve all
of:

- exact-set gain >= 0.05;
- recall gain >= 0.03;
- precision delta >= -0.02;
- strictly lower mean absolute cardinality error.

If no learned decoder passes, retain F0 and do not run A100 replication. If a
decoder passes, freeze the highest-ranked advancing arm before replicating it
and the frozen control on one A100. The ranking/hard-negative branch remains
unrun during this campaign.
