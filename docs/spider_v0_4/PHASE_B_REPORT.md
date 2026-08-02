# Phase B: renderer causal test

## Decision

The orthogonal-aligned renderer does **not** advance. It won zero of three
matched seeds against the historical independent-renderer control under the
registered lookup, reachability, exact-set, and precision gates. Phase C is
therefore not authorised by this result.

All nine evaluations completed without crashes. The six B1/B2 runs trained new
models; B0 reused the immutable v0.3 1,000-step checkpoints. Every run had zero
deterministic replay mismatches, zero row-permutation decision mismatches, and
zero sealed access.

## Aggregate development result

| Arm | Exact set | Precision | Recall | Scored coverage | Macro AP | FP/case |
|---|---:|---:|---:|---:|---:|---:|
| B0 independent control | 0.5866 | 0.6534 | 0.4720 | 0.9779 | 0.9082 | 0.1348 |
| B1 shared additive | 0.5036 | 0.9471 | 0.1660 | 0.9853 | 0.8860 | 0.0085 |
| B2 orthogonal aligned | 0.5924 | 0.8743 | 0.3936 | 0.9662 | 0.8710 | 0.0599 |

B2 improves mean precision substantially relative to B0 but does not clear the
0.90 precision or 0.98 scored-positive-coverage constraints in aggregate. Its
small +0.0059 exact-set difference is far below the +0.10 gate, while recall is
lower by 0.0785.

## Matched B2 deltas from B0

| Seed | Lookup recall | Reachability recall | Exact set | Precision | Gate |
|---:|---:|---:|---:|---:|---|
| 1701 | -0.0156 | -0.2422 | -0.1201 | +0.1754 | fail |
| 1802 | +0.0078 | +0.3281 | +0.2695 | +0.2776 | fail |
| 1903 | -0.0078 | -0.6719 | -0.1318 | +0.2097 | fail |

The sign reversal across seeds rules out a stable renderer-only gain. B2's
lookup recall is 0.0000, 0.0078, and 0.0000. The model-selection procedure also
chooses steps 500, 750, and 500, confirming that simply extending these runs is
not supported by the registered evidence.

## Initial diagnosis

The failure is not candidate reachability alone. In many families, positives
are scored and macro AP remains high, but the precision-constrained calibrated
threshold becomes extremely conservative. B1/B2 raw evidence thresholds range
from roughly 0.64 to 0.99; five of six aligned runs use at least 0.94. This
suppresses nearly all lookup evidence and often corroboration evidence.

At the same time, lookup ranking is not perfect: mean worst-positive ranks are
about 1.5 and lookup macro AP is roughly 0.74-0.78. The aligned renderer is
identifiable to a bilinear probe, but the pooled scorer only receives symmetric
mean/max summaries followed by an MLP. It has no explicit cross-set matching
operation, so Phase A's information-theoretic result does not imply that this
particular baseline will exploit the geometry.

The next registered action is a mechanical generator/label audit plus frozen
oracle-cardinality and per-case-threshold ceilings. Those diagnostics will
determine whether to repair labels, run the set-decoding branch, or test the
dedicated/slot-aware readouts. No scaling or Phase C run is justified yet.

Machine-readable evidence is under
`artifacts/spider_v0_4/phase_b/local_rtx5070ti/`. Raw checkpoints and complete
per-case reports remain local and content-addressed; aggregate summaries and
the experiment ledger are versioned.
