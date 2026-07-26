"""Verify one extracted isolated Spider v0.1 Colab run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SOURCE_COMMIT = "966bccab778e5e4ba4b50b74ed14b9c038df8746"
DATASET_DIGEST = (
    "101af9fd4ff38a9b8416675fb57941d8a9d99126f4f80e2c406c0087287a3105"
)
EXPERIMENT_ID = re.compile(
    r"L-E00[45]-(?:recurrent|pooled)-s(?:1701|1802|1903)-5k"
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run_directory = args.run_directory.resolve()
    experiment_id = run_directory.name
    if EXPERIMENT_ID.fullmatch(experiment_id) is None:
        raise ValueError(f"unexpected experiment ID: {experiment_id!r}")

    manifest = read_json(run_directory / "ARTIFACT_MANIFEST.json")
    if manifest.get("experiment_id") != experiment_id:
        raise RuntimeError("artifact manifest experiment ID mismatch")
    if manifest.get("dataset_split_digest") != DATASET_DIGEST:
        raise RuntimeError("artifact manifest dataset drift")
    if manifest.get("source_model_commit") != SOURCE_COMMIT:
        raise RuntimeError("artifact manifest source drift")
    if manifest.get("sealed_access_count") != 0:
        raise RuntimeError("artifact manifest reports sealed access")

    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise TypeError("artifact manifest files must be a mapping")
    actual_files = {
        str(path.relative_to(run_directory))
        for path in run_directory.rglob("*")
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"
    }
    if actual_files != set(manifest_files):
        raise RuntimeError("artifact file set does not match manifest")
    for relative_path, expected in manifest_files.items():
        path = run_directory / relative_path
        if path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"{relative_path}: byte count mismatch")
        if sha256(path) != expected["sha256"]:
            raise RuntimeError(f"{relative_path}: SHA-256 mismatch")

    record = read_json(run_directory / "replication_record.json")
    metrics = read_json(run_directory / "run" / "metrics.json")
    if record.get("experiment_id") != experiment_id:
        raise RuntimeError("replication record experiment ID mismatch")
    if record.get("status") != "accepted":
        raise RuntimeError("replication record is not accepted")
    if record.get("steps") != 5_000:
        raise RuntimeError("replication record step count mismatch")
    if record.get("sealed_access_count") != 0:
        raise RuntimeError("replication record reports sealed access")
    if record.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("replication record source drift")
    if record.get("dataset_split_digest") != DATASET_DIGEST:
        raise RuntimeError("replication record dataset drift")

    if metrics.get("pass") is not True:
        raise RuntimeError("evaluator guard did not pass")
    if metrics.get("sealed_access_count") != 0:
        raise RuntimeError("metrics report sealed access")
    reports = metrics.get("reports")
    if not isinstance(reports, dict):
        raise TypeError("metrics reports must be a mapping")
    if any("sealed" in str(name).lower() for name in reports):
        raise RuntimeError("metrics contain a sealed report")
    if metrics.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("metrics source drift")
    if metrics.get("dataset_split_digest") != DATASET_DIGEST:
        raise RuntimeError("metrics dataset drift")

    checkpoint = run_directory / "run" / "checkpoint.pt"
    checkpoint_sha256 = sha256(checkpoint)
    if checkpoint_sha256 != record.get("checkpoint_sha256"):
        raise RuntimeError("checkpoint hash differs from replication record")
    if checkpoint_sha256 != metrics.get("checkpoint_sha256"):
        raise RuntimeError("checkpoint hash differs from metrics")

    print(
        json.dumps(
            {
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": checkpoint_sha256,
                "experiment_id": experiment_id,
                "score": record["score"],
                "sealed_access_count": 0,
                "verified_file_count": len(actual_files) + 1,
            },
            sort_keys=True,
        )
    )


main()
