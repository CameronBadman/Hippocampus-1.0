from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path("scripts/run_spider_v0_3_autoresearch.py")
    spec = importlib.util.spec_from_file_location(
        "spider_v03_autoresearch",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(
    *,
    recall: float,
    exact: float,
    precision: float = 0.8,
    coverage: float = 0.96,
    passed: bool = True,
):
    return {
        "pass": passed,
        "evidence_gate_metrics": {
            "recall": recall,
            "exact_set_accuracy": exact,
            "precision": precision,
            "scored_positive_coverage": coverage,
            "conditional_selection_recall": 0.9,
            "false_positives_per_case": 0.1,
        },
    }


def test_screen_advances_only_an_arm_winning_two_matched_seeds() -> None:
    module = _module()
    screen = {}
    for seed in module.SEEDS:
        screen[("E0", seed)] = _metrics(recall=0.60, exact=0.40)
    screen[("E1", 1701)] = _metrics(recall=0.66, exact=0.41)
    screen[("E1", 1802)] = _metrics(recall=0.67, exact=0.41)
    screen[("E1", 1903)] = _metrics(recall=0.61, exact=0.40)
    screen[("E2", 1701)] = _metrics(recall=0.68, exact=0.44)
    screen[("E2", 1802)] = _metrics(
        recall=0.70,
        exact=0.45,
        precision=0.70,
    )
    screen[("E2", 1903)] = _metrics(recall=0.61, exact=0.40)

    decision = module._screen_decision(screen)

    assert decision["arms"]["E1"]["eligible"]
    assert not decision["arms"]["E2"]["eligible"]
    assert decision["experimental_winner"] == "E1"
    assert decision["full_arms"] == ["E0", "E1"]


def test_guard_failed_run_cannot_count_as_seed_win() -> None:
    module = _module()
    screen = {}
    for seed in module.SEEDS:
        screen[("E0", seed)] = _metrics(recall=0.60, exact=0.40)
        screen[("E1", seed)] = _metrics(
            recall=0.70,
            exact=0.50,
            passed=seed == 1701,
        )
        screen[("E2", seed)] = _metrics(recall=0.60, exact=0.40)

    decision = module._screen_decision(screen)

    assert decision["arms"]["E1"]["seed_wins"] == 1
    assert decision["experimental_winner"] is None


def test_incomplete_run_is_archived_and_resumed_from_its_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    module = _module()
    output_root = tmp_path / "evidence"
    experiment_id = "V03-screen-E0-s1701-1k"
    incomplete = output_root / "runs" / experiment_id
    incomplete.mkdir(parents=True)
    checkpoint = incomplete / "checkpoint.pt"
    checkpoint.write_bytes(b"exact-resume-state")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "source_commit": "a" * 40,
                }
            )
        )

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    metrics = module._run_or_load(
        arm="E0",
        seed=1701,
        phase="screen",
        output_root=output_root,
        source_commit="a" * 40,
        stop_after_steps=1000,
        timeout_seconds=60,
        precision_floor=0.0,
    )

    recovery = output_root / "recovery" / f"{experiment_id}-attempt-001"
    assert (recovery / "checkpoint.pt").read_bytes() == b"exact-resume-state"
    command = captured["command"]
    assert Path(command[command.index("--resume-checkpoint") + 1]) == (
        recovery / "checkpoint.pt"
    )
    assert metrics["source_commit"] == "a" * 40
    attempt = json.loads(
        (output_root / "attempts.jsonl").read_text().splitlines()[0]
    )
    assert attempt["status"] == "recovered"


def test_incomplete_run_without_checkpoint_is_archived_and_restarted(
    tmp_path,
    monkeypatch,
) -> None:
    module = _module()
    output_root = tmp_path / "evidence"
    experiment_id = "V03-screen-E0-s1701-1k"
    incomplete = output_root / "runs" / experiment_id
    incomplete.mkdir(parents=True)
    (incomplete / "partial.log").write_text("setup failed")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "source_commit": "b" * 40,
                }
            )
        )

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._run_or_load(
        arm="E0",
        seed=1701,
        phase="screen",
        output_root=output_root,
        source_commit="b" * 40,
        stop_after_steps=1000,
        timeout_seconds=60,
        precision_floor=0.0,
    )

    assert "--resume-checkpoint" not in captured["command"]
    assert (
        output_root
        / "recovery"
        / f"{experiment_id}-attempt-001"
        / "partial.log"
    ).is_file()


def test_recovery_prefers_latest_periodic_checkpoint(tmp_path) -> None:
    module = _module()
    output_root = tmp_path / "evidence"
    experiment_id = "V03-full-E1-s1802-6k"
    incomplete = output_root / "runs" / experiment_id
    incomplete.mkdir(parents=True)
    (incomplete / "checkpoint_step_002000.pt").write_bytes(b"step-2k")
    (incomplete / "checkpoint_step_004000.pt").write_bytes(b"step-4k")

    recovered = module._archive_incomplete_output(
        incomplete,
        output_root=output_root,
        experiment_id=experiment_id,
        source_commit="c" * 40,
    )

    assert recovered is not None
    assert recovered.name == "checkpoint_step_004000.pt"
    assert recovered.read_bytes() == b"step-4k"


def test_full_selection_uses_gate_then_representative_seed() -> None:
    module = _module()
    full = {}
    exact_by_seed = {1701: 0.40, 1802: 0.50, 1903: 0.90}
    for seed in module.SEEDS:
        baseline = _metrics(recall=0.60, exact=exact_by_seed[seed])
        baseline.update(
            {
                "experiment_id": f"E0-{seed}",
                "checkpoint_path": f"/runs/E0-{seed}/checkpoint.pt",
                "checkpoint_sha256": str(seed) * 16,
                "calibration": {
                    "selected": {
                        "raw_probability_threshold": 0.4,
                    }
                },
                "dataset_hash": "d" * 64,
                "source_commit": "c" * 40,
            }
        )
        candidate = _metrics(
            recall=0.66 if seed != 1903 else 0.61,
            exact=exact_by_seed[seed] + 0.01,
        )
        candidate.update(
            {
                "experiment_id": f"E1-{seed}",
                "checkpoint_path": f"/runs/E1-{seed}/checkpoint.pt",
                "checkpoint_sha256": str(seed + 1) * 16,
                "calibration": {
                    "selected": {
                        "raw_probability_threshold": 0.45,
                    }
                },
                "dataset_hash": "d" * 64,
                "source_commit": "c" * 40,
            }
        )
        full[("E0", seed)] = baseline
        full[("E1", seed)] = candidate

    decision = module._final_evidence_selection(
        full,
        experimental_arm="E1",
    )

    assert decision["experimental_full_passed"]
    assert decision["selected_arm"] == "E1"
    # 1802 is the median-performing seed, not the cherry-picked best seed.
    assert decision["selected_seed"] == 1802


def test_failed_full_gate_falls_back_to_e0() -> None:
    module = _module()
    full = {}
    for seed in module.SEEDS:
        for arm, recall in (("E0", 0.60), ("E2", 0.61)):
            metrics = _metrics(recall=recall, exact=0.4)
            metrics.update(
                {
                    "experiment_id": f"{arm}-{seed}",
                    "checkpoint_path": f"/{arm}-{seed}.pt",
                    "checkpoint_sha256": "a" * 64,
                    "calibration": {
                        "selected": {
                            "raw_probability_threshold": 0.5,
                        }
                    },
                    "dataset_hash": "d" * 64,
                    "source_commit": "c" * 40,
                }
            )
            full[(arm, seed)] = metrics

    decision = module._final_evidence_selection(
        full,
        experimental_arm="E2",
    )

    assert not decision["experimental_full_passed"]
    assert decision["selected_arm"] == "E0"
