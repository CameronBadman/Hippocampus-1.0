"""Run one frozen Spider v0.2 recurrence experiment on A100/H100 Colab.

The Better Colab launch source injects a validated ``RUN_SPEC`` mapping before
executing this file. One session runs one model/seed pair so provider loss
cannot destroy the rest of the matrix.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from IPython.display import display


SOURCE_COMMIT = "acb533666d481daf9b6fb56562d69a5dd78c5e0e"
REPOSITORY_URL = "https://github.com/CameronBadman/Hippocampus-1.0.git"
REPOSITORY = Path("/content/hippocampus-spider-v02-source")
OUTPUT_ROOT = Path("/content/spider-v02-recurrence")
ALLOWED_ACCELERATORS = ("H100", "A100")
ALLOWED_SEEDS = (1701, 1802, 1903)
ALLOWED_MODELS = ("recurrent", "pooled")
STEPS = 6_000
TRAINING_TIMEOUT_SECONDS = 14_400
TRAIN_MANIFEST_SHA256 = (
    "ff36529a8090581f6156a8fc36258e4a14eee9a542955623b70550001469fe56"
)
VALIDATION_MANIFEST_SHA256 = (
    "67c2273e4899af179bc1e10185742b806d751f5f5dba858c771f2eca8a6af4aa"
)
CONFIGS = {
    "recurrent": "configs/spider_v0_2/recurrent_recurrence.json",
    "pooled": "configs/spider_v0_2/pooled_recurrence.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )


def validate_spec(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise TypeError("RUN_SPEC must be a mapping")
    model = str(raw.get("model"))
    seed = int(raw.get("seed", -1))
    experiment_id = str(raw.get("experiment_id"))
    if model not in ALLOWED_MODELS:
        raise ValueError(f"unsupported model: {model!r}")
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"unsupported seed: {seed}")
    expected = f"REC-{model}-s{seed}-6k"
    if experiment_id != expected:
        raise ValueError(
            f"experiment ID {experiment_id!r} does not match {expected!r}"
        )
    return {
        "model": model,
        "seed": seed,
        "experiment_id": experiment_id,
    }


def write_status(
    output: Path,
    spec: dict[str, object],
    *,
    state: str,
    error: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    payload = {
        "experiment_id": spec["experiment_id"],
        "model": spec["model"],
        "seed": spec["seed"],
        "state": state,
        "error": error,
        "sealed_access_count": 0,
        "timestamp": utc_now(),
    }
    if extra:
        payload.update(extra)
    output.mkdir(parents=True, exist_ok=True)
    (output / "JOB_STATUS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _verify_repository_checkout() -> None:
    resolved = run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        capture=True,
    ).stdout.strip()
    if resolved != SOURCE_COMMIT:
        raise RuntimeError(f"source checkout drifted to {resolved}")
    status = run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY,
        capture=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("source checkout contains tracked modifications")


def prepare_repository() -> str:
    if REPOSITORY.exists():
        _verify_repository_checkout()
        return "reused_verified_checkout"
    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            REPOSITORY_URL,
            str(REPOSITORY),
        ],
        timeout=300,
    )
    run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=REPOSITORY)
    _verify_repository_checkout()
    run(
        [sys.executable, "-m", "pip", "install", "--quiet", "-e", ".[test]"],
        cwd=REPOSITORY,
        timeout=900,
    )
    return "fresh_clone_and_install"


def verify_environment(
    output: Path,
    *,
    repository_preparation: str,
) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("training requires a visible CUDA accelerator")
    device = torch.cuda.get_device_name(0)
    if not any(name in device.upper() for name in ALLOWED_ACCELERATORS):
        raise RuntimeError(f"requires H100/A100, received {device!r}")
    driver = run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture=True,
    ).stdout.strip()
    tests = run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY,
        timeout=1_200,
        capture=True,
    )
    split_index = json.loads(
        (
            REPOSITORY
            / "artifacts/spider_v0_2/splits/MANIFEST_INDEX.json"
        ).read_text()
    )
    manifests = split_index["manifests"]
    if (
        manifests["train_recurrence_necessity"]["sha256"]
        != TRAIN_MANIFEST_SHA256
    ):
        raise RuntimeError("training manifest hash drift")
    if (
        manifests["validation_recurrence_necessity"]["sha256"]
        != VALIDATION_MANIFEST_SHA256
    ):
        raise RuntimeError("validation manifest hash drift")
    if split_index["sealed_split_present"]:
        raise RuntimeError("development dataset unexpectedly contains sealed data")
    environment = {
        "cuda_device": device,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver_and_memory": driver,
        "python": sys.version,
        "repository_preparation": repository_preparation,
        "torch": torch.__version__,
        "tests": tests.stdout.strip().splitlines()[-1],
    }
    (output / "ENVIRONMENT.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )
    return environment


def monitor_training(
    process: subprocess.Popen[bytes],
    *,
    output: Path,
    log_path: Path,
    spec: dict[str, object],
    timeout_seconds: int,
) -> tuple[int, dict[str, str]]:
    started = time.monotonic()
    observed_checkpoints: dict[str, tuple[int, int]] = {}
    published_checkpoints: dict[str, str] = {}
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            process.kill()
            process.wait(timeout=30)
            raise TimeoutError(
                f"training exceeded {timeout_seconds} seconds"
            )
        checkpoint_paths = sorted(
            (output / "run").glob("checkpoint*.pt")
        )
        current_observations = {
            path.name: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in checkpoint_paths
        }
        for path in checkpoint_paths:
            if (
                path.name not in published_checkpoints
                and observed_checkpoints.get(path.name)
                == current_observations[path.name]
            ):
                published_checkpoints[path.name] = publish_binary_artifact(
                    path,
                    experiment_id=str(spec["experiment_id"]),
                    purpose="periodic_checkpoint",
                )
        observed_checkpoints = current_observations
        utilization = run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture=True,
        ).stdout.strip()
        heartbeat = {
            "elapsed_seconds": round(elapsed, 1),
            "experiment_id": spec["experiment_id"],
            "checkpoints": sorted(current_observations),
            "published_checkpoints": published_checkpoints,
            "gpu_utilization_percent_and_memory_mib": utilization,
            "log_bytes": log_path.stat().st_size if log_path.exists() else 0,
        }
        write_status(
            output,
            spec,
            state="training",
            extra=heartbeat,
        )
        print(json.dumps({"heartbeat": heartbeat}, sort_keys=True), flush=True)
        time.sleep(60)
    for path in sorted((output / "run").glob("checkpoint*.pt")):
        if path.name not in published_checkpoints:
            published_checkpoints[path.name] = publish_binary_artifact(
                path,
                experiment_id=str(spec["experiment_id"]),
                purpose="final_checkpoint",
            )
    return int(process.returncode or 0), published_checkpoints


def train_one(
    spec: dict[str, object],
    output: Path,
) -> dict[str, object]:
    run_output = output / "run"
    log_path = output / "training.log"
    arguments = [
        sys.executable,
        "-u",
        "scripts/train_spider_recurrence.py",
        "--config",
        CONFIGS[str(spec["model"])],
        "--experiment-id",
        str(spec["experiment_id"]),
        "--output-dir",
        str(run_output),
        "--seed",
        str(spec["seed"]),
        "--steps",
        str(STEPS),
    ]
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            arguments,
            cwd=REPOSITORY,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return_code, published_checkpoints = monitor_training(
        process,
        output=output,
        log_path=log_path,
        spec=spec,
        timeout_seconds=TRAINING_TIMEOUT_SECONDS,
    )
    if return_code:
        tail = log_path.read_text(errors="replace")[-20_000:]
        raise RuntimeError(
            f"training exited {return_code}; log tail:\n{tail}"
        )
    metrics = json.loads((run_output / "metrics.json").read_text())
    if metrics["status"] != "accepted" or not metrics["pass"]:
        raise RuntimeError("training evaluator rejected the run")
    if metrics["sealed_access_count"] != 0:
        raise RuntimeError("training evaluator accessed sealed data")
    if metrics["source_commit"] != SOURCE_COMMIT:
        raise RuntimeError("training evaluator source commit drift")
    if metrics["resolved_steps"] != STEPS:
        raise RuntimeError("training step budget drift")
    checkpoints = sorted(run_output.glob("checkpoint*.pt"))
    expected_names = {
        "checkpoint.pt",
        *{
            f"checkpoint_step_{step:06d}.pt"
            for step in range(1_000, STEPS, 1_000)
        },
    }
    if {path.name for path in checkpoints} != expected_names:
        raise RuntimeError("periodic checkpoint set is incomplete")
    checkpoint_records = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in checkpoints
    }
    expected_published = {
        name: str(metadata["sha256"])
        for name, metadata in checkpoint_records.items()
    }
    if published_checkpoints != expected_published:
        raise RuntimeError("published checkpoint hashes are incomplete")
    return {
        "experiment_id": spec["experiment_id"],
        "model": spec["model"],
        "seed": spec["seed"],
        "steps": STEPS,
        "status": "accepted",
        "failure_reason": None,
        "sealed_access_count": 0,
        "source_commit": SOURCE_COMMIT,
        "runtime_seconds": metrics["runtime_seconds"],
        "training_runtime_seconds": metrics["training"]["runtime_seconds"],
        "parameter_count": metrics["parameter_count"],
        "primary_structural_success": metrics[
            "primary_structural_success"
        ],
        "primary_final_autonomous_success": metrics[
            "primary_final_autonomous_success"
        ],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "checkpoints": checkpoint_records,
        "published_checkpoints": published_checkpoints,
        "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
    }


def package_result(
    spec: dict[str, object],
    output: Path,
    record: dict[str, object],
) -> tuple[Path, str]:
    (output / "experiment_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "experiment_id": spec["experiment_id"],
        "source_commit": SOURCE_COMMIT,
        "sealed_access_count": 0,
        "file_count_excluding_manifest": len(files),
        "files": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    (output / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    archive = Path(
        shutil.make_archive(
            f"/content/{spec['experiment_id']}-result",
            "zip",
            root_dir=output.parent,
            base_dir=output.name,
        )
    )
    return archive, sha256(archive)


def publish_archive(archive: Path, archive_sha256: str) -> None:
    encoded = base64.b64encode(archive.read_bytes()).decode("ascii")
    display(
        {"application/zip": encoded},
        metadata={
            "filename": archive.name,
            "sha256": archive_sha256,
        },
        raw=True,
    )


def publish_binary_artifact(
    path: Path,
    *,
    experiment_id: str,
    purpose: str,
) -> str:
    artifact_sha256 = sha256(path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    display(
        {"application/octet-stream": encoded},
        metadata={
            "experiment_id": experiment_id,
            "filename": path.name,
            "purpose": purpose,
            "sha256": artifact_sha256,
        },
        raw=True,
    )
    return artifact_sha256


def main() -> None:
    injected = globals().get("RUN_SPEC")
    spec = validate_spec(injected)
    experiment_id = str(spec["experiment_id"])
    output = OUTPUT_ROOT / experiment_id
    if output.exists():
        raise FileExistsError(f"refusing to reuse {output}")
    write_status(output, spec, state="setup")
    record: dict[str, object]
    error: str | None = None
    try:
        repository_preparation = prepare_repository()
        environment = verify_environment(
            output,
            repository_preparation=repository_preparation,
        )
        write_status(
            output,
            spec,
            state="training",
            extra={"environment": environment},
        )
        record = train_one(spec, output)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        record = {
            "experiment_id": experiment_id,
            "model": spec["model"],
            "seed": spec["seed"],
            "steps": STEPS,
            "status": "failed",
            "failure_reason": error,
            "sealed_access_count": 0,
            "source_commit": SOURCE_COMMIT,
        }
    write_status(
        output,
        spec,
        state="packaging",
        error=error,
    )
    archive, archive_sha256 = package_result(spec, output, record)
    write_status(
        output,
        spec,
        state="finished" if error is None else "failed",
        error=error,
        extra={
            "archive": str(archive),
            "archive_sha256": archive_sha256,
            "checkpoint_sha256": record.get("checkpoint_sha256"),
        },
    )
    publish_archive(archive, archive_sha256)
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "status": record["status"],
                "archive": str(archive),
                "archive_sha256": archive_sha256,
                "checkpoint_sha256": record.get("checkpoint_sha256"),
                "primary_structural_success": record.get(
                    "primary_structural_success"
                ),
                "primary_final_autonomous_success": record.get(
                    "primary_final_autonomous_success"
                ),
                "sealed_access_count": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if error is not None:
        raise RuntimeError(error)


main()
