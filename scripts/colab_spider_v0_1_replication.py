"""Run the pre-registered post-sealed Spider v0.1 Colab replications.

This file is submitted as one exact, durable better-colab execution. It clones
the frozen source commit, runs tests, trains six matched non-sealed experiments,
and emits one immutable ZIP artifact per run plus an aggregate records bundle.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from IPython.display import display


SOURCE_COMMIT = "966bccab778e5e4ba4b50b74ed14b9c038df8746"
DATASET_DIGEST = (
    "101af9fd4ff38a9b8416675fb57941d8a9d99126f4f80e2c406c0087287a3105"
)
REPOSITORY_URL = "https://github.com/CameronBadman/Hippocampus-1.0.git"
REPOSITORY = Path("/content/hippocampus-spider-v01")
OUTPUT = Path("/content/spider-v01-colab-replication")
STEPS = 2_000
SEEDS = (701, 802, 903)
RUNS = tuple(
    (
        f"C-E004-recurrent-s{seed}-2k",
        "artifacts/spider_v0_1/autoresearch/configs/"
        "E004-hierarchical-recurrent.json",
        seed,
        "recurrent",
    )
    for seed in SEEDS
) + tuple(
    (
        f"C-E005-pooled-s{seed}-2k",
        "artifacts/spider_v0_1/autoresearch/configs/"
        "E005-hierarchical-pooled.json",
        seed,
        "pooled",
    )
    for seed in SEEDS
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


def publish_zip(source: Path, name: str) -> None:
    archive = Path(
        shutil.make_archive(
            str(OUTPUT / name),
            "zip",
            root_dir=source,
        )
    )
    encoded = base64.b64encode(archive.read_bytes()).decode("ascii")
    display(
        {"application/zip": encoded},
        metadata={
            "filename": archive.name,
            "sha256": sha256(archive),
        },
        raw=True,
    )
    archive.unlink()


def prepare_repository() -> None:
    if not REPOSITORY.exists():
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
    run(["git", "fetch", "origin", SOURCE_COMMIT], cwd=REPOSITORY, timeout=120)
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


def verify_environment() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Colab replication requires a visible CUDA device")
    tests = run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY,
        timeout=600,
        capture=True,
    )
    index = json.loads(
        (
            REPOSITORY
            / "artifacts/spider_v0_1/splits/MANIFEST_INDEX.json"
        ).read_text()
    )
    if index["aggregate_sha256"] != DATASET_DIGEST:
        raise RuntimeError("dataset manifest hash drift")
    if index["sealed_cases_materialised"]:
        raise RuntimeError("frozen source says sealed cases were materialised")
    return {
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "driver": run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture=True,
        ).stdout.strip(),
        "python": sys.version,
        "tests": tests.stdout.strip().splitlines()[-1],
        "torch": torch.__version__,
    }


def materialize_config(
    run_id: str,
    base_path: str,
    seed: int,
) -> Path:
    config = json.loads((REPOSITORY / base_path).read_text())
    config["name"] = run_id
    config["training"]["device"] = "cuda"
    config["training"]["dtype"] = "float32"
    config["training"]["steps"] = STEPS
    config["training"]["seed"] = seed
    config["training"]["log_every"] = 100
    config["replication"] = {
        "analysis_status": "post-sealed; no selection effect",
        "dataset_split_digest": DATASET_DIGEST,
        "source_commit": SOURCE_COMMIT,
    }
    path = OUTPUT / "configs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return path


def train_one(
    run_id: str,
    base_path: str,
    seed: int,
    model_family: str,
) -> dict[str, object]:
    run_output = OUTPUT / "runs" / run_id
    run_output.mkdir(parents=True, exist_ok=False)
    config = materialize_config(run_id, base_path, seed)
    log_path = run_output / "process.log"
    started = time.perf_counter()
    status = "accepted"
    failure_reason = None
    try:
        completed = run(
            [
                sys.executable,
                "scripts/spider_v0_1_evaluator.py",
                "--config",
                str(config),
                "--experiment-id",
                run_id,
                "--output-dir",
                str(run_output),
            ],
            cwd=REPOSITORY,
            timeout=2_400,
            capture=True,
        )
        log_path.write_text(completed.stdout + completed.stderr)
        metrics = json.loads((run_output / "metrics.json").read_text())
        if metrics["sealed_access_count"] != 0:
            raise RuntimeError("replication attempted sealed access")
        if any("sealed" in name for name in metrics["reports"]):
            raise RuntimeError("replication report contains a sealed split")
        if metrics["dataset_split_digest"] != DATASET_DIGEST:
            raise RuntimeError("run reported a different dataset digest")
    except Exception as error:
        status = "failed"
        failure_reason = f"{type(error).__name__}: {error}"
        if not log_path.exists():
            log_path.write_text(failure_reason + "\n")
        metrics = {
            "config": json.loads(config.read_text()),
            "dataset_split_digest": DATASET_DIGEST,
            "experiment_id": run_id,
            "failure_reason": failure_reason,
            "seed": seed,
            "source_commit": SOURCE_COMMIT,
        }

    record = {
        **metrics,
        "colab_replication": {
            "analysis_status": "post-sealed; no selection effect",
            "model_family": model_family,
            "replication_runtime_seconds": time.perf_counter() - started,
            "status": status,
        },
        "failure_reason": failure_reason or metrics.get("failure_reason"),
        "status": status,
    }
    (run_output / "colab_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    publish_zip(run_output, run_id)
    print(
        json.dumps(
            {
                "experiment_id": run_id,
                "score": record.get("score"),
                "status": status,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return record


def write_summary(
    records: list[dict[str, object]],
    environment: dict[str, object],
) -> None:
    ledger = OUTPUT / "colab_replications.jsonl"
    with ledger.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    accepted = [
        record
        for record in records
        if record["status"] == "accepted"
    ]
    groups: dict[str, list[float]] = {"recurrent": [], "pooled": []}
    for record in accepted:
        family = record["colab_replication"]["model_family"]
        groups[family].append(float(record["score"]))
    aggregate = {
        "analysis_status": "post-sealed; no selection effect",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset_split_digest": DATASET_DIGEST,
        "environment": environment,
        "groups": {
            family: {
                "mean": statistics.fmean(scores) if scores else None,
                "population_stddev": (
                    statistics.pstdev(scores) if len(scores) > 1 else 0.0
                ),
                "scores": scores,
            }
            for family, scores in groups.items()
        },
        "run_count": len(records),
        "source_commit": SOURCE_COMMIT,
    }
    (OUTPUT / "COLAB_REPLICATION_SUMMARY.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Spider v0.1 Colab replication",
        "",
        "These runs occurred after sealed evaluation and cannot change model "
        "selection.",
        "",
        "| Model | Scores | Mean | Population SD |",
        "|---|---|---:|---:|",
    ]
    for family, values in aggregate["groups"].items():
        scores = values["scores"]
        lines.append(
            f"| {family} | {', '.join(f'{score:.4f}' for score in scores)} "
            f"| {values['mean'] or 0.0:.4f} "
            f"| {values['population_stddev'] or 0.0:.4f} |"
        )
    (OUTPUT / "COLAB_REPLICATION_SUMMARY.md").write_text(
        "\n".join(lines) + "\n"
    )
    records_bundle = OUTPUT / "records"
    records_bundle.mkdir()
    for path in (
        ledger,
        OUTPUT / "COLAB_REPLICATION_SUMMARY.json",
        OUTPUT / "COLAB_REPLICATION_SUMMARY.md",
    ):
        shutil.copy2(path, records_bundle / path.name)
    shutil.copytree(OUTPUT / "configs", records_bundle / "configs")
    publish_zip(records_bundle, "spider-v01-colab-records")
    print(json.dumps(aggregate, sort_keys=True), flush=True)


OUTPUT.mkdir(parents=True, exist_ok=True)
prepare_repository()
environment = verify_environment()
print(json.dumps({"environment": environment}, sort_keys=True), flush=True)
records = [
    train_one(run_id, base_path, seed, model_family)
    for run_id, base_path, seed, model_family in RUNS
]
write_summary(records, environment)
