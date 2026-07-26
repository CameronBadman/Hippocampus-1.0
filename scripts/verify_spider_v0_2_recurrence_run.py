#!/usr/bin/env python3
"""Deeply verify one extracted Spider v0.2 recurrence result archive."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch


SOURCE_COMMIT = "acb533666d481daf9b6fb56562d69a5dd78c5e0e"
DATASET_VERSION = "spider-programs-v0.3-recurrence-dev"
TRAIN_MANIFEST_SHA256 = (
    "ff36529a8090581f6156a8fc36258e4a14eee9a542955623b70550001469fe56"
)
VALIDATION_MANIFEST_SHA256 = (
    "67c2273e4899af179bc1e10185742b806d751f5f5dba858c771f2eca8a6af4aa"
)
EXPERIMENT_ID = re.compile(
    r"REC-(?P<model>recurrent|pooled)-s"
    r"(?P<seed>1701|1802|1903)-6k"
)
EXPECTED_CHECKPOINT_STEPS = (1_000, 2_000, 3_000, 4_000, 5_000)


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


def verify_artifact_manifest(run_directory: Path) -> dict[str, Any]:
    """Verify the manifest covers every extracted file byte-for-byte."""

    manifest_path = run_directory / "ARTIFACT_MANIFEST.json"
    manifest = read_json(manifest_path)
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, Mapping):
        raise TypeError("artifact manifest files must be a mapping")
    actual_files = {
        str(path.relative_to(run_directory)): path
        for path in run_directory.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(actual_files) != set(manifest_files):
        missing = sorted(set(manifest_files).difference(actual_files))
        extra = sorted(set(actual_files).difference(manifest_files))
        raise RuntimeError(
            f"artifact file set mismatch; missing={missing!r}, extra={extra!r}"
        )
    for relative_path, path in actual_files.items():
        expected = manifest_files[relative_path]
        if not isinstance(expected, Mapping):
            raise TypeError(f"{relative_path}: manifest entry is invalid")
        if expected.get("bytes") != path.stat().st_size:
            raise RuntimeError(f"{relative_path}: byte count mismatch")
        if expected.get("sha256") != sha256(path):
            raise RuntimeError(f"{relative_path}: SHA-256 mismatch")
    expected_count = manifest.get("file_count_excluding_manifest")
    if expected_count != len(actual_files):
        raise RuntimeError("artifact manifest file count mismatch")
    return manifest


def _require_zero_sealed_access(
    value: Mapping[str, Any],
    *,
    source: str,
) -> None:
    if value.get("sealed_access_count") != 0:
        raise RuntimeError(f"{source} reports sealed access")
    if any("sealed" in str(key).lower() for key in value.get("reports", {})):
        raise RuntimeError(f"{source} contains a sealed report")


def _expected_checkpoint_names() -> set[str]:
    return {
        "checkpoint.pt",
        *{
            f"checkpoint_step_{step:06d}.pt"
            for step in EXPECTED_CHECKPOINT_STEPS
        },
    }


def _verify_checkpoint_payload(
    path: Path,
    *,
    expected_step: int,
    expected_model: str,
    expected_seed: int,
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name}: checkpoint must be a mapping")
    if payload.get("step") != expected_step:
        raise RuntimeError(f"{path.name}: checkpoint step mismatch")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise TypeError(f"{path.name}: model config is invalid")
    expected_kind = (
        "hierarchical"
        if model_config.get("termination_mode") == "hierarchical"
        else model_config.get("termination_mode")
    )
    if expected_kind != "hierarchical":
        raise RuntimeError(f"{path.name}: terminator config drift")
    loop_config = payload.get("loop_config")
    if not isinstance(loop_config, Mapping):
        raise TypeError(f"{path.name}: loop config is invalid")
    if loop_config.get("seed") != expected_seed:
        raise RuntimeError(f"{path.name}: seed mismatch")
    if loop_config.get("steps") != 6_000:
        raise RuntimeError(f"{path.name}: configured step count drift")
    schedules = loop_config.get("action_schedule")
    if not isinstance(schedules, list) or len(schedules) != 5:
        raise RuntimeError(f"{path.name}: action schedule drift")
    if any(schedule.get("termination") != 0.0 for schedule in schedules):
        raise RuntimeError(f"{path.name}: learned stopping was enabled")
    policy = payload.get("execution_policy")
    if not isinstance(policy, Mapping):
        raise TypeError(f"{path.name}: execution policy is invalid")
    if policy.get("horizon_mode") != "oracle_required":
        raise RuntimeError(f"{path.name}: horizon policy drift")
    if policy.get("path_state_intervention") != "none":
        raise RuntimeError(f"{path.name}: training state intervention drift")
    loss_config = payload.get("loss_config")
    if not isinstance(loss_config, Mapping):
        raise TypeError(f"{path.name}: loss config is invalid")
    if loss_config.get("termination") != 0.0:
        raise RuntimeError(f"{path.name}: termination loss was enabled")
    if expected_model == "pooled" and any(
        "path_seed" in key for key in payload["model"]
    ):
        raise RuntimeError(f"{path.name}: pooled model identity drift")


def verify_run(run_directory: Path) -> dict[str, Any]:
    """Deeply verify one extracted run and return a compact identity."""

    run_directory = run_directory.resolve()
    match = EXPERIMENT_ID.fullmatch(run_directory.name)
    if match is None:
        raise ValueError(f"unexpected experiment ID: {run_directory.name!r}")
    experiment_id = run_directory.name
    model = match.group("model")
    seed = int(match.group("seed"))

    manifest = verify_artifact_manifest(run_directory)
    if manifest.get("experiment_id") != experiment_id:
        raise RuntimeError("artifact manifest experiment ID mismatch")
    if manifest.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("artifact manifest source drift")
    _require_zero_sealed_access(manifest, source="artifact manifest")

    record = read_json(run_directory / "experiment_record.json")
    metrics = read_json(run_directory / "run" / "metrics.json")
    job_status = read_json(run_directory / "JOB_STATUS.json")
    environment = read_json(run_directory / "ENVIRONMENT.json")
    checkpoint_manifest = read_json(
        run_directory / "run" / "checkpoint.manifest.json"
    )
    for name, value in (
        ("experiment record", record),
        ("metrics", metrics),
        ("job status", job_status),
    ):
        _require_zero_sealed_access(value, source=name)

    if record.get("experiment_id") != experiment_id:
        raise RuntimeError("experiment record ID mismatch")
    if record.get("model") != model or record.get("seed") != seed:
        raise RuntimeError("experiment record model/seed drift")
    if record.get("status") != "accepted":
        raise RuntimeError("experiment record is not accepted")
    if record.get("failure_reason") is not None:
        raise RuntimeError("accepted experiment record has a failure")
    if record.get("steps") != 6_000:
        raise RuntimeError("experiment record step count drift")
    if record.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("experiment record source drift")

    if metrics.get("experiment_id") != experiment_id:
        raise RuntimeError("metrics experiment ID mismatch")
    if metrics.get("status") != "accepted" or metrics.get("pass") is not True:
        raise RuntimeError("metrics guard rejected the run")
    if metrics.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("metrics source drift")
    if metrics.get("dataset_version") != DATASET_VERSION:
        raise RuntimeError("metrics dataset version drift")
    if metrics.get("resolved_seed") != seed:
        raise RuntimeError("metrics seed drift")
    if metrics.get("resolved_steps") != 6_000:
        raise RuntimeError("metrics step count drift")
    dataset_hashes = metrics.get("dataset_hashes")
    if not isinstance(dataset_hashes, Mapping):
        raise TypeError("metrics dataset hashes are invalid")
    if (
        dataset_hashes.get("train_recurrence_necessity")
        != TRAIN_MANIFEST_SHA256
    ):
        raise RuntimeError("training manifest hash drift")
    if (
        dataset_hashes.get("validation_recurrence_necessity")
        != VALIDATION_MANIFEST_SHA256
    ):
        raise RuntimeError("validation manifest hash drift")
    guard = metrics.get("guard")
    if not isinstance(guard, Mapping):
        raise TypeError("metrics guard is invalid")
    if guard.get("finite") is not True:
        raise RuntimeError("metrics contain non-finite values")
    for field in (
        "deterministic_replay_mismatches",
        "row_permutation_mismatches",
        "sealed_access_count",
    ):
        if guard.get(field) != 0:
            raise RuntimeError(f"metrics guard failed: {field}")

    if job_status.get("experiment_id") != experiment_id:
        raise RuntimeError("job status experiment ID mismatch")
    if job_status.get("state") not in {"packaging", "finished"}:
        raise RuntimeError("job status did not reach packaging")
    if job_status.get("error") is not None:
        raise RuntimeError("job status contains an error")

    device = str(environment.get("cuda_device", "")).upper()
    if "A100" not in device and "H100" not in device:
        raise RuntimeError(f"unapproved accelerator: {device!r}")
    tests = str(environment.get("tests", ""))
    if "passed" not in tests:
        raise RuntimeError("remote substrate test result is missing")

    checkpoint_paths = {
        path.name: path
        for path in (run_directory / "run").glob("checkpoint*.pt")
    }
    if set(checkpoint_paths) != _expected_checkpoint_names():
        raise RuntimeError("checkpoint set is incomplete")
    checkpoint_records = record.get("checkpoints")
    if not isinstance(checkpoint_records, Mapping):
        raise TypeError("experiment checkpoint records are invalid")
    published = record.get("published_checkpoints")
    if not isinstance(published, Mapping):
        raise TypeError("published checkpoint records are invalid")
    if set(checkpoint_records) != set(checkpoint_paths):
        raise RuntimeError("experiment checkpoint record set mismatch")
    if set(published) != set(checkpoint_paths):
        raise RuntimeError("published checkpoint set mismatch")

    identities: dict[str, dict[str, Any]] = {}
    for name, path in checkpoint_paths.items():
        expected_step = (
            6_000
            if name == "checkpoint.pt"
            else int(name.removeprefix("checkpoint_step_").removesuffix(".pt"))
        )
        digest = sha256(path)
        metadata = checkpoint_records[name]
        if not isinstance(metadata, Mapping):
            raise TypeError(f"{name}: checkpoint record is invalid")
        if metadata.get("bytes") != path.stat().st_size:
            raise RuntimeError(f"{name}: record byte count mismatch")
        if metadata.get("sha256") != digest:
            raise RuntimeError(f"{name}: record SHA-256 mismatch")
        if published.get(name) != digest:
            raise RuntimeError(f"{name}: published SHA-256 mismatch")
        _verify_checkpoint_payload(
            path,
            expected_step=expected_step,
            expected_model=model,
            expected_seed=seed,
        )
        identities[name] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
            "step": expected_step,
        }

    final_checkpoint = checkpoint_paths["checkpoint.pt"]
    final_sha256 = sha256(final_checkpoint)
    if record.get("checkpoint_sha256") != final_sha256:
        raise RuntimeError("record final checkpoint SHA-256 mismatch")
    if metrics.get("checkpoint_sha256") != final_sha256:
        raise RuntimeError("metrics final checkpoint SHA-256 mismatch")
    if checkpoint_manifest.get("checkpoint_sha256") != final_sha256:
        raise RuntimeError("checkpoint manifest SHA-256 mismatch")
    if checkpoint_manifest.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("checkpoint manifest source drift")
    if checkpoint_manifest.get("sealed_access_count") != 0:
        raise RuntimeError("checkpoint manifest reports sealed access")

    return {
        "checkpoints": identities,
        "experiment_id": experiment_id,
        "model": model,
        "seed": seed,
        "sealed_access_count": 0,
        "source_commit": SOURCE_COMMIT,
        "verified_file_count": manifest["file_count_excluding_manifest"] + 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run_directory), sort_keys=True))


if __name__ == "__main__":
    main()
