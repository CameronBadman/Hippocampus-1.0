"""Compare cached and offsets-derived int32 lengths on CUDA.

Run with:

    python benchmarks/benchmark_cached_lengths.py

The benchmark isolates the structural preparation shared by gather and layout
construction. It reports CUDA-event medians after warm-up and applies the
project's 5% retain/regression rule.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class Workload:
    name: str
    owner_count: int
    selected_count: int
    minimum_length: int
    maximum_length: int


WORKLOADS = (
    Workload("many_small", 65_536, 32_768, 0, 8),
    Workload("fewer_long", 8_192, 4_096, 32, 224),
)


def _measure_cuda_ms(
    operation: Callable[[], tuple[torch.Tensor, ...]],
    *,
    warmup: int = 40,
    repeats: int = 200,
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        # Retain the outputs until after the event so allocation and kernels
        # cannot be optimized away by Python reference lifetime.
        del outputs
    return statistics.median(samples)


def _run_workload(workload: Workload) -> dict[str, float | str | int]:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(20260726)
    lengths = torch.randint(
        workload.minimum_length,
        workload.maximum_length + 1,
        (workload.owner_count,),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.cumsum(lengths.to(torch.int64), dim=0),
        )
    ).to(torch.int32)
    selected = torch.randint(
        0,
        workload.owner_count,
        (workload.selected_count,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )

    def cached_path() -> tuple[torch.Tensor, ...]:
        selected_lengths = lengths[selected]
        selected_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.cumsum(selected_lengths.to(torch.int64), dim=0),
            )
        )
        nonempty = torch.nonzero(selected_lengths > 0).flatten()
        max_length = selected_lengths.max()
        return selected_lengths, selected_offsets, nonempty, max_length

    def derived_path() -> tuple[torch.Tensor, ...]:
        derived_lengths = offsets[1:] - offsets[:-1]
        selected_lengths = derived_lengths[selected]
        selected_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.cumsum(selected_lengths.to(torch.int64), dim=0),
            )
        )
        nonempty = torch.nonzero(selected_lengths > 0).flatten()
        max_length = selected_lengths.max()
        return selected_lengths, selected_offsets, nonempty, max_length

    cached_ms = _measure_cuda_ms(cached_path)
    derived_ms = _measure_cuda_ms(derived_path)
    improvement = (derived_ms - cached_ms) / derived_ms * 100
    return {
        "workload": workload.name,
        "owners": workload.owner_count,
        "selected": workload.selected_count,
        "cached_median_ms": cached_ms,
        "derived_median_ms": derived_ms,
        "cached_improvement_percent": improvement,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    results = [_run_workload(workload) for workload in WORKLOADS]
    improvements = [
        float(result["cached_improvement_percent"]) for result in results
    ]
    retain = all(improvement >= 5.0 for improvement in improvements) and all(
        improvement >= -5.0 for improvement in improvements
    )
    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "results": results,
        "decision": "retain_cached_lengths" if retain else "derive_from_offsets",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

