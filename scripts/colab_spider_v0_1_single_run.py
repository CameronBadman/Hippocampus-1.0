"""Run and package one frozen Spider v0.1 5k replication on Colab.

Each invocation is intentionally limited to one model/seed pair so a provider
session loss cannot destroy the rest of the matrix. The caller supplies a
small JSON specification and downloads the resulting archive before releasing
the session.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from typing import Any


SOURCE_COMMIT = "966bccab778e5e4ba4b50b74ed14b9c038df8746"
DATASET_DIGEST = (
    "101af9fd4ff38a9b8416675fb57941d8a9d99126f4f80e2c406c0087287a3105"
)
REPOSITORY_URL = "https://github.com/CameronBadman/Hippocampus-1.0.git"
REPOSITORY = Path("/content/hippocampus-spider-v01-source")
OUTPUT_ROOT = Path("/content/spider-v01-colab-5k-isolated")
STEPS = 5_000
ALLOWED_ACCELERATORS = ("H100", "A100")
ALLOWED_SEEDS = (1701, 1802, 1903)
BASE_CONFIGS = {
    "pooled": (
        "artifacts/spider_v0_1/autoresearch/configs/"
        "E005-hierarchical-pooled.json"
    ),
    "recurrent": (
        "artifacts/spider_v0_1/autoresearch/configs/"
        "E004-hierarchical-recurrent.json"
    ),
}


@dataclass(frozen=True, slots=True)
class RunSpec:
    model_family: str
    seed: int

    @property
    def experiment_id(self) -> str:
        number = "004" if self.model_family == "recurrent" else "005"
        return f"L-E{number}-{self.model_family}-s{self.seed}-5k"

    @classmethod
    def from_path(cls, path: Path) -> RunSpec:
        raw = json.loads(path.read_text())
        model_family = str(raw["model_family"])
        seed = int(raw["seed"])
        if model_family not in BASE_CONFIGS:
            raise ValueError(f"unsupported model family: {model_family!r}")
        if seed not in ALLOWED_SEEDS:
            raise ValueError(f"unregistered seed: {seed}")
        expected_id = cls(model_family, seed).experiment_id
        if raw.get("experiment_id") != expected_id:
            raise ValueError("experiment ID does not match model family/seed")
        return cls(model_family=model_family, seed=seed)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(
    output: Path,
    *,
    state: str,
    spec: RunSpec,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    status = {
        "error": error,
        "experiment_id": spec.experiment_id,
        "sealed_access_count": 0,
        "state": state,
        "timestamp": utc_now(),
    }
    if extra is not None:
        status.update(extra)
    output.mkdir(parents=True, exist_ok=True)
    (output / "JOB_STATUS.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )


def prepare_repository() -> None:
    if REPOSITORY.exists():
        raise FileExistsError(f"refusing to reuse existing {REPOSITORY}")
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
    resolved = run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        capture=True,
    ).stdout.strip()
    if resolved != SOURCE_COMMIT:
        raise RuntimeError(f"source checkout drifted to {resolved}")
    run(
        [sys.executable, "-m", "pip", "install", "--quiet", "-e", ".[test]"],
        cwd=REPOSITORY,
        timeout=600,
    )


def verify_environment(output: Path) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("replication requires a visible CUDA accelerator")
    device_name = torch.cuda.get_device_name(0)
    if not any(name in device_name.upper() for name in ALLOWED_ACCELERATORS):
        raise RuntimeError(f"requires H100/A100, received {device_name!r}")
    driver = run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        capture=True,
    ).stdout.strip()
    tests = run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY,
        timeout=900,
        capture=True,
    )
    split_index = json.loads(
        (
            REPOSITORY
            / "artifacts/spider_v0_1/splits/MANIFEST_INDEX.json"
        ).read_text()
    )
    if split_index["aggregate_sha256"] != DATASET_DIGEST:
        raise RuntimeError("dataset manifest hash drift")
    if split_index["sealed_cases_materialised"]:
        raise RuntimeError("source reports prior sealed materialisation")
    environment = {
        "cuda_device": device_name,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": driver,
        "python": sys.version,
        "test_result": tests.stdout.strip().splitlines()[-1],
        "torch": torch.__version__,
    }
    (output / "ENVIRONMENT.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )
    return environment


def materialize_config(spec: RunSpec, output: Path) -> Path:
    config = json.loads((REPOSITORY / BASE_CONFIGS[spec.model_family]).read_text())
    config["name"] = spec.experiment_id
    config["training"]["device"] = "cuda"
    config["training"]["dtype"] = "float32"
    config["training"]["steps"] = STEPS
    config["training"]["seed"] = spec.seed
    config["training"]["log_every"] = 250
    config["replication"] = {
        "analysis_status": "post-sealed; no selection effect",
        "dataset_split_digest": DATASET_DIGEST,
        "execution_isolation": "one fresh Colab session per run",
        "source_commit": SOURCE_COMMIT,
    }
    path = output / "config.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return path


def validate_metrics(metrics: dict[str, Any], spec: RunSpec) -> None:
    if metrics.get("pass") is not True:
        raise RuntimeError("evaluator guard did not pass")
    if metrics.get("sealed_access_count") != 0:
        raise RuntimeError("evaluator attempted sealed access")
    reports = metrics.get("reports")
    if not isinstance(reports, dict):
        raise TypeError("reports must be a mapping")
    if any("sealed" in str(name).lower() for name in reports):
        raise RuntimeError("evaluator emitted a sealed report")
    if metrics.get("dataset_split_digest") != DATASET_DIGEST:
        raise RuntimeError("evaluator reported dataset drift")
    if metrics.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("evaluator reported source drift")
    if metrics["config"]["name"] != spec.experiment_id:
        raise RuntimeError("evaluator reported experiment ID drift")


def train_one(
    spec: RunSpec,
    output: Path,
    config: Path,
) -> dict[str, Any]:
    run_output = output / "run"
    run_output.mkdir()
    started = time.perf_counter()
    completed = run(
        [
            sys.executable,
            "scripts/spider_v0_1_evaluator.py",
            "--config",
            str(config),
            "--experiment-id",
            spec.experiment_id,
            "--output-dir",
            str(run_output),
        ],
        cwd=REPOSITORY,
        timeout=3_600,
        capture=True,
    )
    (run_output / "process.log").write_text(
        completed.stdout + completed.stderr
    )
    metrics = json.loads((run_output / "metrics.json").read_text())
    validate_metrics(metrics, spec)
    return {
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "config": metrics["config"],
        "dataset_split_digest": metrics["dataset_split_digest"],
        "device": metrics["device"],
        "dtype": metrics["dtype"],
        "experiment_id": spec.experiment_id,
        "failure_reason": None,
        "model_family": spec.model_family,
        "parameter_count": metrics["parameter_count"],
        "peak_memory_bytes": metrics["peak_memory_bytes"],
        "reports": metrics["reports"],
        "runtime_seconds": metrics["runtime_seconds"],
        "score": metrics["score"],
        "sealed_access_count": 0,
        "seed": spec.seed,
        "source_commit": metrics["source_commit"],
        "status": "accepted",
        "steps": STEPS,
        "training": metrics["training"],
        "worker_runtime_seconds": time.perf_counter() - started,
    }


def package_result(
    spec: RunSpec,
    output: Path,
    record: dict[str, Any],
) -> tuple[Path, str]:
    (output / "replication_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "dataset_split_digest": DATASET_DIGEST,
        "experiment_id": spec.experiment_id,
        "file_count_excluding_manifest": len(files),
        "files": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
        "sealed_access_count": 0,
        "source_model_commit": SOURCE_COMMIT,
    }
    (output / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    archive = Path(f"/content/{spec.experiment_id}-result.tar.gz")
    if archive.exists():
        raise FileExistsError(f"refusing to replace existing {archive}")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output, arcname=spec.experiment_id)
    return archive, sha256(archive)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    spec = RunSpec.from_path(args.spec)
    output = OUTPUT_ROOT / spec.experiment_id
    if output.exists():
        raise FileExistsError(f"refusing to reuse existing {output}")
    write_status(output, state="setup", spec=spec)
    try:
        prepare_repository()
        environment = verify_environment(output)
        config = materialize_config(spec, output)
        write_status(
            output,
            state="training",
            spec=spec,
            extra={"environment": environment},
        )
        record = train_one(spec, output, config)
        write_status(
            output,
            state="packaging",
            spec=spec,
            extra={"score": record["score"]},
        )
        archive, archive_sha256 = package_result(spec, output, record)
        write_status(
            output,
            state="finished",
            spec=spec,
            extra={
                "archive": str(archive),
                "archive_sha256": archive_sha256,
                "score": record["score"],
            },
        )
        print(
            json.dumps(
                {
                    "archive": str(archive),
                    "archive_sha256": archive_sha256,
                    "experiment_id": spec.experiment_id,
                    "score": record["score"],
                    "sealed_access_count": 0,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException as error:
        write_status(
            output,
            state="failed",
            spec=spec,
            error=f"{type(error).__name__}: {error}",
        )
        raise


main()
