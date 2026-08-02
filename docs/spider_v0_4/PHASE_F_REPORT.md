# Spider v0.4 Phase F1: Learned Evidence Set Decoding

## Outcome

No learned set decoder passed the preregistered matched-seed gate. F0, the
immutable pooled/global-threshold control, remains selected. A100 replication
was conditional on a learned arm advancing and was therefore not run.

| Arm | Policy | Exact set | Precision | Recall | Coverage | Macro AP | Cardinality MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F0 | calibrated global threshold | 0.7549 | 0.8830 | 0.7516 | 0.9992 | 0.9378 | 0.2682 |
| F1 | learned null boundary | 0.7633 | 0.8688 | 0.7700 | 0.9936 | 0.9236 | 0.2881 |
| F2 | learned cardinality + top-k | 0.4805 | 0.5902 | 0.4932 | 0.9996 | 0.9319 | 0.4600 |
| F3 | null + cardinality + top-k | 0.5687 | 0.8773 | 0.4135 | 0.9996 | 0.9395 | 0.4456 |

The primary constraint was precision >= 0.90 with scored-positive coverage >=
0.98. F0, F2, and F3 met it on zero of three seeds; F1 met it on one. All
three learned arms won zero of three matched-seed advancement gates.

F1 is the only arm close to the control. Its mean deltas were `+0.0085` exact
set, `+0.0184` recall, `-0.0143` precision, and `+0.0199` cardinality error.
These miss both required gains and do not reduce cardinality error. F2 and F3
are decisive negative results.

## Family result

Exact-set accuracy and recall expose different decoder failure modes:

| Family | Metric | F0 | F1 | F2 | F3 |
| --- | --- | ---: | ---: | ---: | ---: |
| lookup | exact set | 0.5013 | 0.5091 | 0.4531 | 0.5104 |
| lookup | recall | 0.0339 | 0.0469 | 0.4010 | 0.0703 |
| reachability | exact set | 0.6185 | 0.6120 | 0.5508 | 0.6224 |
| reachability | recall | 0.4141 | 0.4714 | 0.2578 | 0.2448 |
| latest-valid | exact set | 0.9779 | 0.9909 | 0.4648 | 0.6810 |
| latest-valid | recall | 0.9961 | 1.0000 | 0.6862 | 0.6224 |
| corroboration | exact set | 0.9219 | 0.9414 | 0.4531 | 0.4609 |
| corroboration | recall | 0.9781 | 0.9948 | 0.4698 | 0.4510 |

F2 substantially raises lookup recall but does so by forcing many wrong
selections: lookup precision is only 0.3247, and overall false positives rise
to 0.2819 per case. F3's null boundary suppresses false positives but also
suppresses required evidence, especially for latest-valid and corroboration.
The learned total-cardinality signal therefore does not reproduce the oracle
cardinality ceiling.

## Interpretation

Phase E established a real *diagnostic* set-decoding ceiling: correct
cardinality applied to frozen ranked candidates raises D0 exact set from
0.7549 to 0.9274. Phase F shows that this ceiling is not currently deployable.
The global state readout does not learn a sufficiently reliable cardinality or
per-case boundary from the registered training states. High AP alone is not
enough: lookup ranking remains imperfect, and errors in predicted count turn a
top-k policy into false positives or false negatives across all later rounds.

The next experiment should not enlarge Spider. A bounded follow-up should
first measure the cardinality head itself by round, family, answerability, and
available candidate count, then train the decoder on a family/cardinality-
balanced frozen-state dataset. Only if that head predicts held-out cardinality
accurately should it be reattached to the controller. The clean hard-negative
ranking branch remains available later, but this result does not authorise
simultaneous decoder and ranking changes.

## Integrity

- New accepted training runs: 9
- Historical F0 observations reused: 3
- Seeds: 1701, 1802, 1903
- Maximum steps: 2,000; checkpoint interval: 250
- Deterministic replay mismatches: 0 for every run
- Row-permutation decision mismatches: 0 for every run
- Sealed accesses: 0

One F1/1802 attempt was invalidated when two orchestration sessions were found
writing the same directory. Both were stopped, all partial checkpoints were
quarantined, and the seed was rerun from step zero under a new filesystem
campaign lock. The invalid attempt remains in the failure ledger and is not
included in any result.

Machine-readable results, the JSONL ledger, generated Markdown ledger, gate
decision, and selected-control manifest are under
`artifacts/spider_v0_4/phase_f/local_rtx5070ti/`.
