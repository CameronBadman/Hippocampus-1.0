# Spider v0.4 Phase D evidence-readout result

## Outcome

All 15 registered 1,000-step screens completed on the local RTX 5070 Ti with
seeds 1701, 1802, and 1903. No readout ablation passed the preregistered
matched-seed gate. D0 (pooled/shared) and D2 (Spider/shared) therefore remain
the pooled and Spider controls and are the only arms advanced to 2,000 steps.

| Arm | Processor/readout | Exact set | Precision | Recall | Coverage | Macro AP |
|---|---|---:|---:|---:|---:|---:|
| D0 | pooled/shared | 0.7588 | 0.8993 | 0.7460 | 0.9976 | 0.9216 |
| D1 | pooled/dedicated | 0.7607 | 0.9222 | 0.7348 | 0.9996 | 0.9270 |
| D2 | Spider/shared | 0.7422 | 0.9046 | 0.7111 | 0.9920 | 0.8947 |
| D3 | Spider/dedicated | 0.7425 | 0.8978 | 0.7175 | 0.9972 | 0.9118 |
| D4 | Spider/slot-aware | 0.7516 | 0.9203 | 0.7196 | 0.9992 | 0.9167 |

D1 versus D0, D3 versus D2, and D4 versus D2 each won zero of three
matched-seed gates. D4 also failed to clear D3 on any seed. Dedicated heads
mostly improve precision or AP; neither separating the evidence head nor
reading path rows directly provides the required exact-set or recall effect.

## Family diagnosis

The aggregate result hides a sharp family split. Mean recall by family is:

| Arm | Lookup | Reachability | Latest-valid | Corroboration |
|---|---:|---:|---:|---:|
| D0 | 0.0026 | 0.3750 | 0.9961 | 0.9917 |
| D1 | 0.0078 | 0.3359 | 0.9831 | 0.9865 |
| D2 | 0.0078 | 0.1693 | 0.9922 | 0.9844 |
| D3 | 0.0156 | 0.1953 | 0.9948 | 0.9854 |
| D4 | 0.0052 | 0.2422 | 0.9779 | 0.9896 |

Cross-modal identifiability alone did not make the trained scorer recover
opaque lookup evidence. Latest-valid and corroboration are nearly saturated,
while lookup is almost never selected at the calibrated operating point and
reachability remains weak. The evidence interface is therefore not limited
primarily by shared-head interference or destruction of useful path-row state.

The frozen-logit Phase B ceiling already routed the downstream experiment to
set decoding. Phase D strengthens that decision: ranking AP is materially
higher than deployed lookup recall, and none of the cleaner readouts solves
selection.

## Integrity and operational note

Every accepted result used dataset hash
`8ff3c7f12978e8381552eafadbe5fc6dfab8eb08c2484204e1cdad7835dc8a32`.
Scored-positive coverage was at least 0.992 in aggregate for every arm. All
deterministic replay and row-permutation decision mismatch counts were zero.
No sealed data was accessed.

D4/1701 timed out once after training and all checkpoint selections because
the monolithic process exceeded 300 seconds during calibration. The timeout is
retained in `failed_experiments.jsonl`. Calibration/evaluation resumed from
the selected checkpoint through the documented bounded-stage mechanism; the
record names both source commits. Later D4 runs used that mechanism from the
start. No model state, selection rule, or data was changed.
