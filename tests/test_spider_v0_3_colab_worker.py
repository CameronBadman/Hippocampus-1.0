from __future__ import annotations

import ast
import importlib.util
import subprocess
from pathlib import Path

import pytest


WORKER = Path("scripts/colab_spider_v0_3_worker.py")


def _module():
    spec = importlib.util.spec_from_file_location(
        "spider_v03_colab_worker",
        WORKER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _constants() -> dict[str, object]:
    tree = ast.parse(WORKER.read_text())
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


def test_worker_requires_one_a100_and_pinned_nonsealed_protocol() -> None:
    constants = _constants()

    assert constants["REQUIRED_ACCELERATOR"] == "A100"
    assert len(str(constants["DATASET_SHA256"])) == 64
    assert constants["HEARTBEAT_SECONDS"] <= 60
    assert constants["TRAINING_TIMEOUT_SECONDS"] >= 21_600
    assert constants["EXPECTED_TERMINATION_RECORDS"] == 9
    assert "spider-v0.2" not in str(constants["DRIVE_PROJECT_PATH"])
    assert "Spider-v0.3-Evidence" in str(constants["DRIVE_PROJECT_PATH"])
    source = WORKER.read_text()
    assert "run_spider_v0_3_autoresearch.py" in source
    assert "run_spider_v0_3_termination_matrix.py" in source


def test_run_spec_requires_exact_commit_and_matrix_protocol() -> None:
    worker = _module()
    valid = {
        "source_commit": "a" * 40,
        "dataset_sha256": worker.DATASET_SHA256,
        "phase": "all",
    }

    assert worker.validate_spec(valid) == valid
    with pytest.raises(ValueError, match="source commit"):
        worker.validate_spec({**valid, "source_commit": "not-a-commit"})
    with pytest.raises(ValueError, match="dataset hash"):
        worker.validate_spec({**valid, "dataset_sha256": "b" * 64})
    with pytest.raises(ValueError, match="phase"):
        worker.validate_spec({**valid, "phase": "summarize"})


def test_stable_sync_waits_one_observation_and_copies_atomically(
    tmp_path,
) -> None:
    worker = _module()
    local = tmp_path / "local"
    remote = tmp_path / "drive"
    checkpoint = local / "runs" / "run-1" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    observed = worker.observe_files(local)
    synced = worker.sync_stable_files(
        local,
        remote,
        previous={},
        copied={},
        force=False,
    )
    assert synced.copied == {}
    assert not (remote / "runs" / "run-1" / "checkpoint.pt").exists()

    synced = worker.sync_stable_files(
        local,
        remote,
        previous=observed,
        copied={},
        force=False,
    )
    copied = remote / "runs" / "run-1" / "checkpoint.pt"
    assert copied.read_bytes() == b"checkpoint"
    assert not copied.with_name("checkpoint.pt.part").exists()
    assert synced.copied["runs/run-1/checkpoint.pt"]["sha256"] == (
        worker.sha256(checkpoint)
    )


def test_drive_restore_never_writes_the_historical_v02_folder(
    tmp_path,
) -> None:
    worker = _module()
    remote = tmp_path / "drive"
    local = tmp_path / "local"
    (remote / "RUN_PROTOCOL.json").parent.mkdir(parents=True)
    (remote / "RUN_PROTOCOL.json").write_text("{}")
    restored = worker.restore_drive_state(remote, local)

    assert restored == 1
    assert (local / "RUN_PROTOCOL.json").is_file()
    source = WORKER.read_text()
    assert "1A8QnvZKDSWeiTXvi6RwYx76LFVAcDZRw" not in source


def test_launch_renderer_pins_head_and_worker_hash() -> None:
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/render_spider_v0_3_colab_launch.py",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # The worker is intentionally still uncommitted in this test-first
    # worktree. Once committed, the exact same command must emit a guarded
    # source cell rather than accepting a dirty worker.
    if completed.returncode:
        assert "uncommitted changes" in completed.stderr
    else:
        source = completed.stdout
        assert "RUN_SPEC" in source
        assert "EXPECTED_SHA256" in source
        assert "worker hash mismatch" in source
