# Generator, label, and frozen-policy audit

## Supervisor result

The Phase B failure is not caused by corrupt oracle labels.

Across all 10,240 registered cases:

- invalid cases: 0;
- evidence-label mismatches: 0;
- required evidence nodes sharing legitimate query-visible identity: 1.000;
- answerable lookup cases solved by an exact observable relation/value/gate
  rule: 1.000 across 1,280 cases;
- ordinary topology/size metadata advantage over the answerability majority
  baseline: 0.000.

This means the positive lookup target is present, observable, and labelled
consistently. Phase A showed that A2 makes its cross-modal symbols recoverable
to a bilinear probe. The pooled scorer's failure to exploit that information is
therefore a model/training-interface result, not a missing-label result.

## Dataset interface defect

The audit did find a separate correctness issue. `UNKNOWN_UNSUPPORTED` cases
append one fresh `unsupported_*` query atom. Every such symbol is unique (1,700
symbols across 1,700 cases), so there is no reusable semantic code for
unsupportedness. The extra row simultaneously makes query cardinality
predictive: a family-plus-query-row-count baseline reaches 0.666 accuracy
against a 0.500 majority baseline, an advantage of 0.166.

That violates the registered requirement that manifold cardinality not predict
correctness. The v0.4 dataset and Phase B result remain immutable. Subsequent
training must use a versioned amendment that removes unsupported cases from the
evidence-specific campaign (learned termination is explicitly deferred) and
restores the cardinality diagnostic to chance.

This defect does not explain the answerable lookup failure: recall is computed
over required positive evidence, and the answerable lookup labels pass the
observable rule exactly.

## Frozen-policy ceilings

The development logits from every Phase B checkpoint were evaluated without
training another model.

| Arm | Global threshold exact | Oracle-cardinality exact | Gain | Per-case threshold exact |
|---|---:|---:|---:|---:|
| B0 | 0.5866 | 0.9105 | +0.3239 | 0.9105 |
| B1 | 0.5036 | 0.8861 | +0.3825 | 0.8861 |
| B2 | 0.5924 | 0.8822 | +0.2897 | 0.8822 |

B2 gains are +0.3447, +0.1904, and +0.3340 across the three matched seeds, so
all three clear the registered +0.15 set-decoding branch threshold. The global
threshold is a major bottleneck.

It is not the only bottleneck. For B2 seed 1701, oracle-cardinality exact set is
0.773 on lookup and 0.824 on reachability (including zero-evidence cases),
rather than 1.000. Positive lookup has roughly one-in-two top-1 recovery. Thus
cardinality-aware decoding can recover much of the aggregate loss, while a
dedicated or slot-aware readout is still justified to improve candidate
ranking.

## Decision

- Preserve and reject the Phase B renderer-only hypothesis.
- Skip Phase C under its preregistered gate.
- Version-correct the unsupported-query interface before more training.
- Continue to the evidence-readout diagnostic on the corrected A2 data.
- Retain the set-decoding branch as unlocked, but train it only after the best
  pooled and Spider readouts are frozen, as originally registered.

Machine-readable audits are in `artifacts/spider_v0_4/diagnostics/`. No sealed
data was accessed.
