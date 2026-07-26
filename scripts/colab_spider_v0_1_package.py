"""Validate and package the completed Spider v0.1 Colab replication.

This script is intentionally fail-closed: it will not create an archive unless
all six frozen runs completed, every evaluator guard passed, and no run
accessed or reported a sealed split.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any


OUTPUT = Path("/content/spider-v01-colab-5k")
ARCHIVE = Path("/content/spider-v01-colab-5k-results.tar.gz")
SOURCE_COMMIT = "966bccab778e5e4ba4b50b74ed14b9c038df8746"
DATASET_DIGEST = (
    "101af9fd4ff38a9b8416675fb57941d8a9d99126f4f80e2c406c0087287a3105"
)
EXPECTED_RUNS = {
    f"L-E00{model_number}-{model}-s{seed}-5k"
    for seed in (1701, 1802, 1903)
    for model_number, model in ((4, "recurrent"), (5, "pooled"))
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_metrics(path: Path) -> None:
    metrics = read_json(path)
    run_id = path.parent.name
    if metrics.get("pass") is not True:
        raise RuntimeError(f"{run_id}: evaluator guard did not pass")
    if metrics.get("sealed_access_count") != 0:
        raise RuntimeError(f"{run_id}: sealed access count was not zero")
    reports = metrics.get("reports")
    if not isinstance(reports, dict):
        raise TypeError(f"{run_id}: reports must be a mapping")
    if any("sealed" in str(name).lower() for name in reports):
        raise RuntimeError(f"{run_id}: sealed report was emitted")
    if metrics.get("dataset_split_digest") != DATASET_DIGEST:
        raise RuntimeError(f"{run_id}: dataset digest drift")
    if metrics.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError(f"{run_id}: source commit drift")


def main() -> None:
    job_status = read_json(OUTPUT / "JOB_STATUS.json")
    if job_status.get("state") != "finished":
        raise RuntimeError(f"job is not finished: {job_status.get('state')!r}")
    completed = set(job_status.get("completed_runs", ()))
    if completed != EXPECTED_RUNS:
        raise RuntimeError(
            f"completed run set mismatch: {sorted(completed)!r}"
        )

    metric_paths = sorted((OUTPUT / "runs").glob("*/metrics.json"))
    metric_run_ids = {path.parent.name for path in metric_paths}
    if metric_run_ids != EXPECTED_RUNS:
        raise RuntimeError(
            f"metric run set mismatch: {sorted(metric_run_ids)!r}"
        )
    for path in metric_paths:
        validate_metrics(path)

    summary = read_json(OUTPUT / "COLAB_5K_SUMMARY.json")
    if summary.get("run_count") != len(EXPECTED_RUNS):
        raise RuntimeError("aggregate run count mismatch")
    if summary.get("dataset_split_digest") != DATASET_DIGEST:
        raise RuntimeError("aggregate dataset digest drift")
    if summary.get("source_model_commit") != SOURCE_COMMIT:
        raise RuntimeError("aggregate source commit drift")

    ledger_lines = [
        line
        for line in (OUTPUT / "colab_5k_experiments.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    if len(ledger_lines) != len(EXPECTED_RUNS):
        raise RuntimeError("ledger record count mismatch")
    ledger_ids: set[str] = set()
    for line in ledger_lines:
        record = json.loads(line)
        ledger_ids.add(record["experiment_id"])
        if record.get("status") != "accepted":
            raise RuntimeError("ledger contains a non-accepted record")
        if record.get("sealed_access_count") != 0:
            raise RuntimeError("ledger contains sealed access")
    if ledger_ids != EXPECTED_RUNS:
        raise RuntimeError("ledger run set mismatch")

    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file())
    manifest = {
        "archive_root": OUTPUT.name,
        "dataset_split_digest": DATASET_DIGEST,
        "file_count_excluding_manifest": len(files),
        "files": {
            str(path.relative_to(OUTPUT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
        "run_ids": sorted(EXPECTED_RUNS),
        "sealed_access_count": 0,
        "source_model_commit": SOURCE_COMMIT,
    }
    manifest_path = OUTPUT / "ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    if ARCHIVE.exists():
        raise FileExistsError(f"refusing to replace existing {ARCHIVE}")
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(OUTPUT, arcname=OUTPUT.name)

    print(
        json.dumps(
            {
                "archive": str(ARCHIVE),
                "bytes": ARCHIVE.stat().st_size,
                "run_count": len(EXPECTED_RUNS),
                "sealed_access_count": 0,
                "sha256": sha256(ARCHIVE),
            },
            sort_keys=True,
        )
    )


main()
