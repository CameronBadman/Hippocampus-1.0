# Spider v0 final report

## Outcome

Spider v0 is implemented as a small, inspectable recurrent graph interpreter
over the repository's real packed graph substrate. It has a working exact
program generator, frozen exchangeable renderer, trainable pooled/flat/recurrent
models, deterministic sparse controller, CPU and CUDA paths, staged
oracle/mixed training, tests, tiny-overfit evidence, ID/OOD reports, and a
machine-readable AutoResearch ledger.

The milestone demonstrates that the data flow and supervised transition
problem are learnable. It does **not** demonstrate reliable autonomous graph
reasoning or a clear recurrent OOD advantage.

## Verification snapshot

- 112 tests passed: 106 CPU tests and 6 CUDA tests.
- Existing packed-substrate tests remained green.
- CUDA FP32 and BF16 forward/backward tests passed.
- RTX 5070 Ti driver/runtime/BF16 capability were verified.
- Final 48-case CUDA tiny overfit reached 1.000 on every supervised action and
  termination metric.
- Repeated deterministic controller execution produced zero mismatched traces.
- Seeded manifold-row permutations produced zero decision mismatches.
- Training/model traversal calls packed manifold gathers and
  `PackedTopology.expand_frontier`; it does not build Python adjacency in the
  hot path.

## Architecture experiment result

The pre-registered matrix used eight primary-seed comparisons and two extra
seeds for each of the top two configurations, consuming all 12 accepted runs.
There were no hidden or discarded failures.

| Primary-seed model | Score |
|---|---:|
| tied recurrent, standard attention | 0.5046 |
| one-hypothesis recurrent controller | 0.5046 |
| recurrent, compositional attention | 0.4796 |
| no context VOI/read budget | 0.4792 |
| untied recurrent | 0.4738 |
| no global evidence | 0.4713 |
| pooled mean/max baseline | 0.4597 |
| flat position-free Transformer | 0.4247 |

The standard recurrent model's three-seed score was
`0.4679 ± 0.0396` (population standard deviation). The pooled model has only
its pre-registered exploration seed at 0.4597. That difference is too small
and under-replicated to claim the recurrent model wins.

The bounded-multiple-hypothesis comparison was numerically identical to the
one-hypothesis control on this small sample, so H3 remains unresolved.
Compositional attention did not improve the primary score, so H1 is not
supported by the initial evidence.

## Sealed result

The selected tied-standard checkpoint opened the 256-case aggregate sealed set
once:

| Metric | Result |
|---|---:|
| teacher-forced candidate Top-1 | 0.9481 |
| teacher-forced candidate MRR | 0.9741 |
| teacher-forced termination accuracy | 0.4677 |
| autonomous termination accuracy | 0.4609 |
| autonomous exact valid-path rate | 0.7656 |
| answered coverage | 0.6875 |
| risk among answered | 0.5284 |
| false-answer rate | 0.3633 |
| unknown-reason accuracy | 0.2734 |
| autonomous evidence F1 | 0.0232 |
| mean arcs scored | 3.6680 |
| mean contexts read | 0.9297 |
| mean recurrent rounds | 1.0000 |
| mean CUDA latency per query | 10.6 ms |
| peak CUDA memory | 56,162,304 bytes |

The controller's structural traces were valid, but the terminator stopped
after one round on average and the evidence selector recalled almost none of
the exact evidence. Strong teacher-forced ranking therefore did not survive
autonomous rollout.

## Functional swap result

Cross-view edge manifolds were swapped by latent functional alignment for 32
validation cases without swap training. Every edge manifold changed.
Candidate Top-1 moved from 0.9412 to 0.9118 (delta -0.0294); termination
accuracy moved from 0.4595 to 0.4865 (delta +0.0270). The directions disagree
and the sample is small, so H2 remains uncertain.

Swap augmentation was not trained because the frozen 12-accepted-run budget
was already exhausted. The swapping infrastructure and behavioural consistency
loss are implemented, and the report explicitly records those modes as not
run rather than inventing results.

## Acceptance checklist

| Criterion | Status |
|---|---|
| existing substrate tests | passed |
| new CPU tests | passed |
| available CUDA tests | passed |
| near-perfect tiny overfit | passed (1.000) |
| full differentiable CUDA training run | passed |
| real packed expansion/gathers | passed |
| row-permutation invariance | zero decision mismatches |
| deterministic replay | zero trace mismatches |
| pooled versus recurrent comparison | completed |
| ID and seven OOD validation splits | completed in every accepted experiment |
| search/context cost metrics | reported |
| AutoResearch ledger/recommendation | completed |
| sealed evaluation | opened once and recorded |

## Limitations and recommendation

This is a synthetic vector benchmark with short, budgeted training. The OOD
validation samples in architecture search were intentionally small. Metadata
diagnostics do not rule out every shortcut. Equivalent surface re-keying is
not natural language. The exact verifier validates the benchmark definition,
not real-world truth.

Most importantly, the recurrent model did not establish a robust advantage
over pooled features, and autonomous evidence/termination failed on the sealed
set. The recommended next experiment is:

1. train the now-implemented scheduled mixed rollout for longer;
2. directly supervise termination after off-oracle states;
3. increase evidence-ledger recall pressure and conflict cases;
4. use larger validation samples and replicate the pooled control;
5. only then reconsider compositional attention or swap training.

Synthetic benchmark accuracy must not be described as production or
real-world validation.
