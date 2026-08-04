# Spider v0.6 Final Report: Zero-Shot Evidence Selection

## Result

Spider v0.6 improved calibration-free evidence recovery but did not meet its
registered success target. Z0 remains the accepted control because no
experimental arm reached a weakest-metric score of 0.82. Z2 is the best
observed experimental arm at 0.7998, but its precision is only 0.8722, so it is
preserved as a diagnostic rather than promoted as a finalist.

This is genuine symbol-disjoint execution after training: evaluation symbols
never occur in training, and inference fits no temperature, threshold,
cardinality, or family-specific policy. It is not an untrained-model claim.

## Frozen experiment

| Arm | Change from control |
| --- | --- |
| Z0 | Global learned NULL; calibration removed |
| Z1 | NULL observes the current candidate set |
| Z2 | Z1 plus graph-balanced candidate-vs-NULL BCE |
| Z3 | Z2 plus a bounded plausible-hard-negative NULL margin |

All arms use the same pooled recurrent processor, renderer, dimensions,
packed-graph execution, fixed horizon, controller budgets, optimizer, and raw
zero-margin inference rule. The campaign completed twelve of twelve registered
runs on an RTX 5070 Ti in FP32 using seeds 1701, 1802, and 1903.

## Aggregate development results

Values are means over three matched seeds. Score is the minimum of exact-set
accuracy, precision, and recall.

| Arm | Score | Exact set | Precision | Recall | Coverage | Macro AP | False positives/case |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Z0 | 0.7600 | 0.7809 | 0.9322 | 0.7600 | 1.0000 | 0.9500 | 0.0462 |
| Z1 | 0.7608 | 0.7910 | 0.9495 | 0.7608 | 0.9992 | 0.9480 | 0.0329 |
| Z2 | 0.7998 | 0.7998 | 0.8722 | 0.8021 | 1.0000 | 0.9492 | 0.1003 |
| Z3 | 0.7937 | 0.7982 | 0.8906 | 0.7937 | 0.9988 | 0.9269 | 0.0830 |

Z1 did not validate the candidate-conditioned-NULL hypothesis by itself: it
won one seed and changed mean recall by only +0.0008. Z2 won two seeds and
raised recall by +0.0421 over Z0, demonstrating that candidate-count imbalance
distorted the learned boundary. That gain purchased too many false positives.
Z3 recovered some lookup positives but reduced the best aggregate score and
macro AP, so it was rejected.

## Program-family diagnosis

Evidence recall by family:

| Arm | Lookup | Reachability | Latest-valid | Corroboration |
| --- | ---: | ---: | ---: | ---: |
| Z0 | 0.0000 | 0.4531 | 0.9935 | 1.0000 |
| Z1 | 0.0000 | 0.4505 | 0.9987 | 0.9990 |
| Z2 | 0.0026 | 0.7161 | 0.9974 | 1.0000 |
| Z3 | 0.0833 | 0.5833 | 0.9961 | 1.0000 |

Z3's lookup effect is not stable enough to accept: lookup precision is 0.2874,
and one of three seeds still has zero lookup recall. Its mean lookup AP is
0.7539, while reachability AP drops materially. The margin redistributes
errors rather than solving cross-modal binding.

Scored-positive coverage is at least 0.9988 for every arm. Candidate
enumeration and packed frontier expansion are therefore not the limiting
stage. The remaining defect lies in evidence scoring/readout and its ability
to preserve query-to-edge-to-destination bindings for unseen symbols.

## Integrity and acceptance decision

- Dataset: `spider-programs-v0.6-zero-shot-dev`
- Aggregate hash: `a05d301bef77d86abcbb658444c2ed277545b82653dac202cef319ecfc1abc17`
- Observable symbol overlap: 0 across 834,560 symbols
- Temperature fits: 0; applied temperature: 1.0
- Sealed accesses: 0
- Deterministic replay mismatches: 0
- Row-permutation decision mismatches: 0
- Full repository tests: 288 passed with CUDA visible

No arm passed the 0.82 target. `FINALIST.json` therefore retains Z0, while
`BEST_OBSERVED.json` records Z2 separately and explicitly marks it unaccepted.
The A100 replication gate did not open.

## Next research decision

Do not add more NULL-loss variants or scale the model. First freeze Z2's
selector and add a lookup-focused state-readout audit on unseen symbols:

1. probe whether query, edge, destination, and path rows retain the correct
   binding before the evidence head;
2. compare the current mean-pooled evidence readout with a dedicated
   slot-aware readout under the same zero-shot rule; and
3. add a direct contrastive binding objective only if the representation is
   present but the evidence logit discards it.

If the binding is absent before readout, repair the synthetic observation and
transition objective rather than the set decoder. If it is present but lost by
mean pooling, the dedicated row-aware head is the smallest justified model
change.

This development-only synthetic result does not validate natural-language
reasoning, real retrieval systems, calibration, or production deployment.
