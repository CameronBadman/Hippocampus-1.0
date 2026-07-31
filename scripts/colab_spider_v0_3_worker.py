"""Run the registered Spider v0.3 evidence matrix on one durable A100.

The launch cell injects ``RUN_SPEC`` before executing this pinned worker.
Stable checkpoints and experiment records are mirrored to a new,
commit-keyed Google Drive directory throughout the run. A replacement Colab
session can restore that directory and resume exact training checkpoints.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, NamedTuple
import zipfile


REPOSITORY_URL = "https://github.com/CameronBadman/Hippocampus-1.0.git"
REPOSITORY = Path("/content/hippocampus-spider-v03-source")
OUTPUT_ROOT = Path("/content/spider-v03-evidence")
DRIVE_PROJECT_PATH = "Hippocampus-1.0/Spider-v0.3-Evidence"
REQUIRED_ACCELERATOR = "A100"
DATASET_SHA256 = (
    "0ed8e27ec44f3773f76b79f1947526f33ba233556b7db91fef04dcb647e5409d"
)
HEARTBEAT_SECONDS = 60
TRAINING_TIMEOUT_SECONDS = 64_800
RUNNER_TIMEOUT_SECONDS = 21_600
EXPECTED_SCREEN_RUNS = 9
EXPECTED_MINIMUM_RECORDS = 12
EXPECTED_MAXIMUM_RECORDS = 15


class SyncResult(NamedTuple):
    observed: dict[str, tuple[int, int]]
    copied: dict[str, dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_spec(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise TypeError("RUN_SPEC must be a mapping")
    source_commit = str(raw.get("source_commit", ""))
    dataset_sha256 = str(raw.get("dataset_sha256", ""))
    phase = str(raw.get("phase", ""))
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")
    if dataset_sha256 != DATASET_SHA256:
        raise ValueError("dataset hash does not match the frozen protocol")
    if phase != "all":
        raise ValueError("phase must be the registered full matrix: 'all'")
    return {
        "source_commit": source_commit,
        "dataset_sha256": dataset_sha256,
        "phase": phase,
    }


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
        timeout=timeout,
        check=True,
        capture_output=capture,
        text=True,
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def observe_files(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.endswith(".part")
    }


def sync_stable_files(
    local_root: Path,
    remote_root: Path,
    *,
    previous: dict[str, tuple[int, int]],
    copied: dict[str, dict[str, Any]],
    force: bool,
) -> SyncResult:
    """Mirror files only after a stable observation, or at final shutdown."""

    current = observe_files(local_root)
    updated = dict(copied)
    for relative, signature in current.items():
        prior_copy = updated.get(relative, {})
        copied_signature = tuple(prior_copy.get("signature", ()))
        stable = force or previous.get(relative) == signature
        if not stable or copied_signature == signature:
            continue
        source = local_root / relative
        destination = remote_root / relative
        _atomic_copy(source, destination)
        digest = sha256(source)
        if sha256(destination) != digest:
            raise RuntimeError(f"Drive hash mismatch after copying {relative}")
        updated[relative] = {
            "bytes": signature[0],
            "mtime_ns": signature[1],
            "sha256": digest,
            "signature": list(signature),
            "synced_at": utc_now(),
        }
    return SyncResult(observed=current, copied=updated)


def restore_drive_state(remote_root: Path, local_root: Path) -> int:
    """Restore an earlier exact-commit mirror without deleting either copy."""

    if not remote_root.exists():
        return 0
    restored = 0
    for source in sorted(remote_root.rglob("*")):
        if not source.is_file() or source.name.endswith(".part"):
            continue
        relative = source.relative_to(remote_root)
        destination = local_root / relative
        _atomic_copy(source, destination)
        restored += 1
    return restored


def mount_drive(source_commit: str) -> Path:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    root = (
        Path("/content/drive/MyDrive")
        / DRIVE_PROJECT_PATH
        / source_commit[:12]
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_status(
    *,
    local_root: Path,
    drive_root: Path | None,
    state: str,
    spec: dict[str, str],
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state": state,
        "error": error,
        "source_commit": spec["source_commit"],
        "dataset_sha256": spec["dataset_sha256"],
        "phase": spec["phase"],
        "sealed_access_count": 0,
        "timestamp": utc_now(),
    }
    if extra:
        payload.update(extra)
    _atomic_json(local_root / "COLAB_JOB_STATUS.json", payload)
    if drive_root is not None:
        _atomic_json(drive_root / "COLAB_JOB_STATUS.json", payload)
    return payload


def _verify_repository_checkout(source_commit: str) -> None:
    resolved = run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        capture=True,
    ).stdout.strip()
    if resolved != source_commit:
        raise RuntimeError(f"source checkout drifted to {resolved}")
    status = run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY,
        capture=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("source checkout contains tracked modifications")


def prepare_repository(source_commit: str) -> str:
    if REPOSITORY.exists():
        _verify_repository_checkout(source_commit)
        return "reused_verified_checkout"
    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            REPOSITORY_URL,
            str(REPOSITORY),
        ],
        timeout=600,
    )
    run(
        ["git", "checkout", "--detach", source_commit],
        cwd=REPOSITORY,
        timeout=120,
    )
    _verify_repository_checkout(source_commit)
    run(
        [sys.executable, "-m", "pip", "install", "--quiet", "-e", ".[test]"],
        cwd=REPOSITORY,
        timeout=1_200,
    )
    return "fresh_clone_and_install"


def verify_environment(
    *,
    source_commit: str,
    repository_preparation: str,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the registered matrix requires a visible A100")
    device = torch.cuda.get_device_name(0)
    if REQUIRED_ACCELERATOR not in device.upper():
        raise RuntimeError(
            f"the registered matrix requires A100, received {device!r}"
        )
    manifest = json.loads(
        (
            REPOSITORY
            / "artifacts/spider_v0_3/splits/MANIFEST_INDEX.json"
        ).read_text()
    )
    if manifest["aggregate_sha256"] != DATASET_SHA256:
        raise RuntimeError("development dataset hash drift")
    if (
        manifest["sealed_access_allowed"]
        or manifest["sealed_cases_materialised"]
        or manifest["sealed_manifest_loaded"]
    ):
        raise RuntimeError("development protocol unexpectedly accessed sealed data")
    tests = run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY,
        timeout=1_800,
        capture=True,
    )
    driver = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture=True,
    ).stdout.strip()
    return {
        "source_commit": source_commit,
        "repository_preparation": repository_preparation,
        "cuda_device": device,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver_and_memory": driver,
        "python": sys.version,
        "torch": torch.__version__,
        "tests": tests.stdout.strip().splitlines()[-1],
    }


def _gpu_status() -> str:
    return run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    ).stdout.strip()


def monitor_matrix(
    process: subprocess.Popen[bytes],
    *,
    spec: dict[str, str],
    drive_root: Path,
    log_path: Path,
) -> dict[str, dict[str, Any]]:
    started = time.monotonic()
    previous: dict[str, tuple[int, int]] = {}
    copied: dict[str, dict[str, Any]] = {}
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed > TRAINING_TIMEOUT_SECONDS:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
            raise TimeoutError(
                f"matrix exceeded {TRAINING_TIMEOUT_SECONDS} seconds"
            )
        sync = sync_stable_files(
            OUTPUT_ROOT,
            drive_root,
            previous=previous,
            copied=copied,
            force=False,
        )
        previous = sync.observed
        copied = sync.copied
        checkpoints = sorted(
            path
            for path in copied
            if path.endswith(".pt")
        )
        heartbeat = {
            "elapsed_seconds": round(elapsed, 1),
            "gpu_utilization_percent_and_memory_mib": _gpu_status(),
            "log_bytes": log_path.stat().st_size if log_path.exists() else 0,
            "drive_file_count": len(copied),
            "drive_checkpoint_count": len(checkpoints),
            "latest_checkpoints": checkpoints[-8:],
            "drive_path": str(drive_root),
        }
        write_status(
            local_root=OUTPUT_ROOT,
            drive_root=drive_root,
            state="training",
            spec=spec,
            extra=heartbeat,
        )
        print(json.dumps({"heartbeat": heartbeat}, sort_keys=True), flush=True)
        time.sleep(HEARTBEAT_SECONDS)
    sync = sync_stable_files(
        OUTPUT_ROOT,
        drive_root,
        previous=previous,
        copied=copied,
        force=True,
    )
    if process.returncode:
        tail = (
            log_path.read_text(errors="replace")[-30_000:]
            if log_path.exists()
            else ""
        )
        raise RuntimeError(
            f"evidence matrix exited {process.returncode}; log tail:\n{tail}"
        )
    return sync.copied


def run_matrix(
    *,
    spec: dict[str, str],
    drive_root: Path,
) -> dict[str, dict[str, Any]]:
    log_path = OUTPUT_ROOT / "colab_matrix.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "scripts/run_spider_v0_3_autoresearch.py",
        "--phase",
        spec["phase"],
        "--output-root",
        str(OUTPUT_ROOT),
        "--timeout-seconds",
        str(RUNNER_TIMEOUT_SECONDS),
    ]
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return monitor_matrix(
        process,
        spec=spec,
        drive_root=drive_root,
        log_path=log_path,
    )


def validate_result(
    *,
    spec: dict[str, str],
    drive_root: Path,
) -> dict[str, Any]:
    ledger_path = OUTPUT_ROOT / "experiments.jsonl"
    records = [
        json.loads(line)
        for line in ledger_path.read_text().splitlines()
        if line.strip()
    ]
    screen_count = sum(record["phase"] == "screen" for record in records)
    if screen_count != EXPECTED_SCREEN_RUNS:
        raise RuntimeError(f"expected 9 screen runs, found {screen_count}")
    if not (
        EXPECTED_MINIMUM_RECORDS
        <= len(records)
        <= EXPECTED_MAXIMUM_RECORDS
    ):
        raise RuntimeError(f"unexpected experiment-record count: {len(records)}")
    for record in records:
        if record["source_commit"] != spec["source_commit"]:
            raise RuntimeError("experiment source commit drift")
        if record["dataset_hash"] != DATASET_SHA256:
            raise RuntimeError("experiment dataset hash drift")
        if record["sealed_access_count"] != 0:
            raise RuntimeError("experiment accessed sealed data")
        if record["status"] not in {"accepted", "guard_violation"}:
            raise RuntimeError(
                f"invalid experiment status: {record['experiment_id']}"
            )
    checkpoint_records: dict[str, dict[str, Any]] = {}
    for checkpoint in sorted(OUTPUT_ROOT.rglob("checkpoint*.pt")):
        relative = str(checkpoint.relative_to(OUTPUT_ROOT))
        remote = drive_root / relative
        if not remote.is_file():
            raise RuntimeError(f"checkpoint missing from Drive: {relative}")
        local_hash = sha256(checkpoint)
        remote_hash = sha256(remote)
        if local_hash != remote_hash:
            raise RuntimeError(f"Drive checkpoint hash mismatch: {relative}")
        checkpoint_records[relative] = {
            "bytes": checkpoint.stat().st_size,
            "sha256": local_hash,
            "drive_path": str(remote),
        }
    if not checkpoint_records:
        raise RuntimeError("matrix produced no checkpoints")
    return {
        "source_commit": spec["source_commit"],
        "dataset_sha256": DATASET_SHA256,
        "sealed_access_count": 0,
        "experiment_record_count": len(records),
        "accepted_run_count": sum(
            record["status"] == "accepted" for record in records
        ),
        "guard_violation_count": sum(
            record["status"] == "guard_violation" for record in records
        ),
        "screen_run_count": screen_count,
        "drive_root": str(drive_root),
        "checkpoint_count": len(checkpoint_records),
        "checkpoints": checkpoint_records,
    }


def package_metadata(spec: dict[str, str]) -> tuple[Path, str]:
    archive = Path(
        f"/content/spider-v03-evidence-{spec['source_commit'][:12]}.zip"
    )
    with zipfile.ZipFile(
        archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as bundle:
        for path in sorted(OUTPUT_ROOT.rglob("*")):
            if path.is_file() and path.suffix != ".pt":
                bundle.write(
                    path,
                    arcname=str(
                        Path(OUTPUT_ROOT.name) / path.relative_to(OUTPUT_ROOT)
                    ),
                )
    return archive, sha256(archive)


def publish_archive(archive: Path, archive_sha256: str) -> None:
    from IPython.display import display

    encoded = base64.b64encode(archive.read_bytes()).decode("ascii")
    display(
        {"application/zip": encoded},
        metadata={
            "filename": archive.name,
            "sha256": archive_sha256,
            "purpose": "Spider v0.3 non-checkpoint experiment records",
        },
        raw=True,
    )


def main() -> None:
    spec = validate_spec(globals().get("RUN_SPEC"))
    drive_root: Path | None = None
    error: str | None = None
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_status(
        local_root=OUTPUT_ROOT,
        drive_root=None,
        state="setup",
        spec=spec,
    )
    try:
        drive_root = mount_drive(spec["source_commit"])
        restored = restore_drive_state(drive_root, OUTPUT_ROOT)
        repository_preparation = prepare_repository(spec["source_commit"])
        environment = verify_environment(
            source_commit=spec["source_commit"],
            repository_preparation=repository_preparation,
        )
        _atomic_json(OUTPUT_ROOT / "COLAB_ENVIRONMENT.json", environment)
        write_status(
            local_root=OUTPUT_ROOT,
            drive_root=drive_root,
            state="training",
            spec=spec,
            extra={
                "restored_file_count": restored,
                "environment": environment,
                "drive_path": str(drive_root),
            },
        )
        run_matrix(spec=spec, drive_root=drive_root)
        drive_manifest = validate_result(
            spec=spec,
            drive_root=drive_root,
        )
        _atomic_json(
            OUTPUT_ROOT / "GOOGLE_DRIVE_CHECKPOINTS.json",
            drive_manifest,
        )
        sync_stable_files(
            OUTPUT_ROOT,
            drive_root,
            previous={},
            copied={},
            force=True,
        )
        archive, archive_sha256 = package_metadata(spec)
        _atomic_copy(archive, drive_root / archive.name)
        status = write_status(
            local_root=OUTPUT_ROOT,
            drive_root=drive_root,
            state="finished",
            spec=spec,
            extra={
                "archive": str(archive),
                "archive_sha256": archive_sha256,
                "drive_path": str(drive_root),
                "checkpoint_count": drive_manifest["checkpoint_count"],
                "experiment_record_count": drive_manifest[
                    "experiment_record_count"
                ],
                "accepted_run_count": drive_manifest["accepted_run_count"],
            },
        )
        publish_archive(archive, archive_sha256)
        print(json.dumps(status, sort_keys=True), flush=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = write_status(
            local_root=OUTPUT_ROOT,
            drive_root=drive_root,
            state="failed",
            spec=spec,
            error=error,
            extra={
                "drive_path": str(drive_root) if drive_root else None,
            },
        )
        if drive_root is not None:
            sync_stable_files(
                OUTPUT_ROOT,
                drive_root,
                previous={},
                copied={},
                force=True,
            )
        print(json.dumps(status, sort_keys=True), flush=True)
    if error is not None:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
