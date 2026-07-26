"""Run the frozen 5,000-step Spider v0.1 H100/A100 replication matrix.

The script is designed for an isolated official Colab CLI job. It clones the
frozen model source, verifies the accelerator and driver, runs the complete
test suite, trains six matched non-sealed experiments, and writes all outputs
under one exportable artifact directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


SOURCE_COMMIT = "966bccab778e5e4ba4b50b74ed14b9c038df8746"
DATASET_DIGEST = (
    "101af9fd4ff38a9b8416675fb57941d8a9d99126f4f80e2c406c0087287a3105"
)
REPOSITORY_URL = "https://github.com/CameronBadman/Hippocampus-1.0.git"
REPOSITORY = Path("/content/hippocampus-spider-v01-source")
OUTPUT = Path("/content/spider-v01-colab-5k")
STEPS = 5_000
SEEDS = (1701, 1802, 1903)
ALLOWED_ACCELERATORS = ("H100", "A100")


@dataclass(frozen=True, slots=True)
class RunSpec:
    experiment_id: str
    base_config: str
    seed: int
    model_family: str


RUNS = tuple(
    spec
    for seed in SEEDS
    for spec in (
        RunSpec(
            experiment_id=f"L-E004-recurrent-s{seed}-5k",
            base_config=(
                "artifacts/spider_v0_1/autoresearch/configs/"
                "E004-hierarchical-recurrent.json"
            ),
            seed=seed,
            model_family="recurrent",
        ),
        RunSpec(
            experiment_id=f"L-E005-pooled-s{seed}-5k",
            base_config=(
                "artifacts/spider_v0_1/autoresearch/configs/"
                "E005-hierarchical-pooled.json"
            ),
            seed=seed,
            model_family="pooled",
        ),
    )
)


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_job_status(
    *,
    state: str,
    completed_runs: list[str],
    active_run: str | None,
    error: str | None = None,
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "JOB_STATUS.json").write_text(
        json.dumps(
            {
                "active_run": active_run,
                "completed_runs": completed_runs,
                "error": error,
                "state": state,
                "timestamp": utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def prepare_repository() -> None:
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
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "-e",
            ".[test]",
        ],
        cwd=REPOSITORY,
        timeout=600,
    )


def verify_environment() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("replication requires a visible CUDA accelerator")
    device_name = torch.cuda.get_device_name(0)
    if not any(name in device_name.upper() for name in ALLOWED_ACCELERATORS):
        raise RuntimeError(
            f"requires H100/A100, received {device_name!r}"
        )
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
    (OUTPUT / "ENVIRONMENT.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )
    return environment


def materialize_config(spec: RunSpec) -> Path:
    config = json.loads((REPOSITORY / spec.base_config).read_text())
    config["name"] = spec.experiment_id
    config["training"]["device"] = "cuda"
    config["training"]["dtype"] = "float32"
    config["training"]["steps"] = STEPS
    config["training"]["seed"] = spec.seed
    config["training"]["log_every"] = 250
    config["replication"] = {
        "analysis_status": "post-sealed; no selection effect",
        "dataset_split_digest": DATASET_DIGEST,
        "source_commit": SOURCE_COMMIT,
    }
    path = OUTPUT / "configs" / f"{spec.experiment_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return path


def train_one(spec: RunSpec) -> dict[str, Any]:
    run_output = OUTPUT / "runs" / spec.experiment_id
    run_output.mkdir(parents=True, exist_ok=False)
    config = materialize_config(spec)
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
    if not metrics["pass"]:
        raise RuntimeError(f"{spec.experiment_id} failed its evaluator guard")
    if metrics["sealed_access_count"] != 0:
        raise RuntimeError(f"{spec.experiment_id} attempted sealed access")
    if any("sealed" in name for name in metrics["reports"]):
        raise RuntimeError(f"{spec.experiment_id} emitted a sealed report")
    if metrics["dataset_split_digest"] != DATASET_DIGEST:
        raise RuntimeError(f"{spec.experiment_id} reported dataset drift")
    record = {
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
    }
    (run_output / "replication_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    record["orchestration_runtime_seconds"] = (
        time.perf_counter() - started
    )
    return record


def write_summary(
    records: list[dict[str, Any]],
    environment: dict[str, Any],
) -> None:
    ledger = OUTPUT / "colab_5k_experiments.jsonl"
    with ledger.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    groups: dict[str, list[float]] = {"recurrent": [], "pooled": []}
    for record in records:
        groups[record["model_family"]].append(float(record["score"]))
    aggregate = {
        "analysis_status": "post-sealed; no selection effect",
        "completed_at": utc_now(),
        "dataset_split_digest": DATASET_DIGEST,
        "environment": environment,
        "groups": {
            family: {
                "mean": statistics.fmean(scores),
                "population_stddev": statistics.pstdev(scores),
                "scores": scores,
            }
            for family, scores in groups.items()
        },
        "run_count": len(records),
        "source_model_commit": SOURCE_COMMIT,
        "steps_per_run": STEPS,
    }
    (OUTPUT / "COLAB_5K_SUMMARY.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Spider v0.1 5k Colab replication",
        "",
        "Post-sealed diagnostic only; these runs cannot change selection.",
        "",
        "| Model | Scores | Mean | Population SD |",
        "|---|---|---:|---:|",
    ]
    for family, values in aggregate["groups"].items():
        scores = values["scores"]
        lines.append(
            f"| {family} | {', '.join(f'{score:.4f}' for score in scores)} "
            f"| {values['mean']:.4f} "
            f"| {values['population_stddev']:.4f} |"
        )
    (OUTPUT / "COLAB_5K_SUMMARY.md").write_text(
        "\n".join(lines) + "\n"
    )


def main() -> None:
    completed_runs: list[str] = []
    write_job_status(
        state="setup",
        completed_runs=completed_runs,
        active_run=None,
    )
    try:
        prepare_repository()
        environment = verify_environment()
        records: list[dict[str, Any]] = []
        for spec in RUNS:
            write_job_status(
                state="training",
                completed_runs=completed_runs,
                active_run=spec.experiment_id,
            )
            record = train_one(spec)
            records.append(record)
            completed_runs.append(spec.experiment_id)
            print(
                json.dumps(
                    {
                        "completed": spec.experiment_id,
                        "score": record["score"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        write_summary(records, environment)
        write_job_status(
            state="finished",
            completed_runs=completed_runs,
            active_run=None,
        )
    except BaseException as error:
        write_job_status(
            state="failed",
            completed_runs=completed_runs,
            active_run=None,
            error=f"{type(error).__name__}: {error}",
        )
        raise


main()
