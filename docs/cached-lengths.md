# Cached manifold-length decision

`PackedManifoldFamily.lengths` remains backed by a cached int32 tensor rather
than deriving `offsets[1:] - offsets[:-1]` at each gather or layout call.

## Decision

Retain cached lengths.

The benchmark rule requires at least a 5% CUDA median-latency improvement in
two representative workloads with no workload regressing by more than 5%.
The benchmark ran on 2026-07-26 with:

- NVIDIA GeForce RTX 5070 Ti;
- PyTorch 2.13.0+cu130; and
- CUDA runtime 13.0.

| Workload | Owners / selected | Cached median | Derived median | Cached improvement |
| --- | ---: | ---: | ---: | ---: |
| Many small manifolds | 65,536 / 32,768 | 0.069968 ms | 0.080512 ms | 13.10% |
| Fewer long manifolds | 8,192 / 4,096 | 0.066048 ms | 0.078960 ms | 16.35% |

Both workloads clear the 5% threshold and neither regresses. The public
`lengths` accessor remains unchanged.

Reproduce with:

```bash
python benchmarks/benchmark_cached_lengths.py
```

CUDA event timing varies with clocks and system load. The decision is based on
the median after warm-up; rerun when the target PyTorch/CUDA stack or dominant
workload changes materially.

