"""Fail-fast CUDA and driver preflight for premium Colab training."""

from __future__ import annotations

import json
import subprocess

import torch


if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see a CUDA device")

device = torch.device("cuda", 0)
device_name = torch.cuda.get_device_name(device)
if "A100" not in device_name.upper() and "H100" not in device_name.upper():
    raise RuntimeError(f"premium accelerator required, received {device_name!r}")

driver = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

left = torch.randn(1024, 1024, device=device)
right = torch.randn(1024, 1024, device=device)
result = left @ right
torch.cuda.synchronize(device)
if not bool(torch.isfinite(result).all().item()):
    raise RuntimeError("CUDA matrix-operation preflight produced non-finite data")

properties = torch.cuda.get_device_properties(device)
print(
    json.dumps(
        {
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
            "cuda_runtime": torch.version.cuda,
            "device": device_name,
            "driver_and_memory_mib": driver,
            "matmul_checksum": float(result.float().mean().item()),
            "torch": torch.__version__,
        },
        sort_keys=True,
    )
)
