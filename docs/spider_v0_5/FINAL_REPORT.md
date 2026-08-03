# Spider v0.5 Final Report: Score Versus Set Decoding

## Result

Spider v0.5 did not find an evidence-selection change that improved recall and
exact evidence recovery while preserving precision. The preregistered control,
X0, remains the selected model.

The negative result is informative. A current-candidate-set count decoder
(X2) raised mean evidence recall from 0.7444 to 0.9075 and mean exact
evidence-set accuracy from 0.7754 to 0.8350. It also reduced precision from
0.9444 to 0.7411 and increased false positives from 0.0358 to 0.2578 per case.
The added positives are therefore real, but the policy does not know reliably
when to select no evidence. An explicit pairwise cross-manifold matcher (X1)
preserved acceptable precision but produced only small, inconsistent gains and
never recovered lookup evidence.

No treatment passed the frozen advancement gate. No sealed split was
materialised or evaluated, and no A100 replication was authorized.

## Frozen experiment

The campaign used a 2x2 factorial:

| Arm | Evidence scorer | Set decoder |
| --- | --- | --- |
| X0 | existing shared pooled head | calibrated global threshold |
| X1 | pairwise query/edge/destination matcher | calibrated global threshold |
| X2 | existing shared pooled head | current-candidate count |
| X3 | pairwise matcher | current-candidate count |

All arms used the same pooled processor, aligned orthogonal renderer,
fixed-horizon controller, data partitions, optimizer settings, and matched
seeds 1701, 1802, and 1903. Each arm trained for at most 2,000 steps with
checkpoint selection every 250 steps. Model selection, calibration, and
development evaluation were disjoint.

- Dataset: `spider-programs-v0.5-score-decode-dev`
- Dataset hash: `b3de9d584148859f4f12a17377005c969372de753677e2b3360fa5e2fa7ada68`
- Frozen run source: `5e832a30db3002e38ff93dc9d6a1c6d9947ca59e`
- Accelerator: NVIDIA GeForce RTX 5070 Ti, FP32
- Accepted runs: 12 of 12
- Sealed accesses: 0

## Aggregate development results

Values are means over the three matched seeds.

| Arm | Exact set | Precision | Recall | Coverage | Macro AP | False positives/case | Cardinality MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| X0 | 0.7754 | 0.9444 | 0.7444 | 0.9996 | 0.9478 | 0.0358 | 0.2422 |
| X1 | 0.7793 | 0.9313 | 0.7584 | 1.0000 | 0.9535 | 0.0462 | 0.2412 |
| X2 | 0.8350 | 0.7411 | 0.9075 | 1.0000 | 0.9468 | 0.2578 | 0.2223 |
| X3 | 0.8252 | 0.8130 | 0.8530 | 1.0000 | 0.9525 | 0.1816 | 0.2236 |

The exact-set main effect of the candidate-count decoder was +0.0527. The
pairwise matcher main effect was -0.0029, and the matcher-by-decoder interaction
was -0.0137. Those effects do not override the precision constraint.

## Program-family diagnosis

Evidence recall by family:

| Arm | Lookup | Reachability | Latest-valid | Corroboration |
| --- | ---: | ---: | ---: | ---: |
| X0 | 0.0000 | 0.3672 | 0.9896 | 0.9969 |
| X1 | 0.0000 | 0.4453 | 0.9948 | 0.9979 |
| X2 | 0.4531 | 0.9557 | 0.9987 | 0.9969 |
| X3 | 0.4089 | 0.6927 | 0.9766 | 0.9958 |

The threshold policy is almost perfect on latest-valid and corroboration but
systematically suppresses lookup and many reachability positives. X2 proves
that those positives are scored and selectable: scored-positive coverage is
1.0 and recall rises sharply. It also selects too much evidence, especially in
reachability. X2 predicts 1.173 evidence items per reachability case on average
against 0.500 required, producing 0.695 false positives per reachability case.

The pairwise matcher does not resolve opaque lookup identity on its own. X1's
lookup recall is zero on all three seeds. X3 is also unstable: seed 1701 retains
0.949 precision but only 0.710 recall, while seeds 1802 and 1903 recover recall
above 0.91 with precision near 0.75.

## Gate decision

An arm had to improve exact-set accuracy by at least 0.05, overall recall by
at least 0.05, and lookup recall by at least 0.20 on two of three matched seeds,
while keeping precision at or above 0.90, coverage at or above 0.98, and
latest-valid/corroboration regressions within 0.02.

- X1: zero seed wins; recall and lookup gates failed.
- X2: zero seed wins; precision failed on every seed.
- X3: zero seed wins; precision or recall failed on every seed, with a
  latest-valid regression on seed 1701.

X0 is retained because no treatment passed, not because X0 solves exact
evidence recovery.

## Correctness and invariance

Every accepted run reported finite metrics, zero deterministic replay
mismatches, zero row-permutation decision mismatches, and zero sealed accesses.
Scored-positive coverage was at least 0.9988 in every run. The result therefore
does not point to candidate enumeration or packed-graph execution as the main
bottleneck.

## Conclusion

The current model has useful evidence-ranking signal, but neither a global
threshold nor an unconstrained count prediction converts it into a precise set.
The immediate bottleneck is a per-round abstention decision: the controller
must distinguish “this candidate set contains useful evidence” from “choose the
best-looking item anyway.” More model capacity is not justified by these data.

The next registered experiment should freeze X0 scoring and test a
candidate-set, ledger-aware `has_evidence_now` gate before conditional ranked
selection. Its null decision must observe the current candidate set and
accumulated evidence, receive direct zero-versus-nonzero supervision, and be
calibrated under the same precision constraint. Candidate count or top-k should
run only after that gate is positive. This isolates false-positive control from
ranking and directly targets the repeated over-selection measured in X2.

This is a synthetic, development-only result. It is not evidence of
production graph retrieval quality, natural-language reasoning, or calibrated
real-world abstention.
