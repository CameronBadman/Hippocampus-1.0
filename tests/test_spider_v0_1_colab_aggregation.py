from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from aggregate_spider_v0_1_colab_runs import (  # noqa: E402
    DATASET_DIGEST,
    SOURCE_COMMIT,
    VALIDATION_SPLITS,
    build_summary,
    render_markdown,
    validate_record,
)


def _record(
    family: str,
    seed: int,
    score: float,
) -> dict[str, object]:
    model_number = 4 if family == "recurrent" else 5
    reports: dict[str, object] = {
        split: {"primary_autonomous_success": score}
        for split in VALIDATION_SPLITS
    }
    reports["validation_id"] = {
        "primary_autonomous_success": score,
        "evidence": {
            "exact_set_accuracy": score,
            "f1": score,
            "precision": score,
            "recall": score,
        },
        "efficiency": {
            "mean_arcs_scored": 3.0,
            "mean_contexts_read": 0.5,
            "mean_rounds": 2.0,
        },
        "rollout": {
            "one_round_stop_rate": 0.25,
            "risk_among_answered": 0.1,
            "termination_accuracy": score,
        },
    }
    return {
        "dataset_split_digest": DATASET_DIGEST,
        "experiment_id": (
            f"L-E00{model_number}-{family}-s{seed}-5k"
        ),
        "model_family": family,
        "reports": reports,
        "score": score,
        "sealed_access_count": 0,
        "seed": seed,
        "source_commit": SOURCE_COMMIT,
        "status": "accepted",
        "steps": 5_000,
    }


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for seed, recurrent, pooled in (
        (1701, 0.4, 0.3),
        (1802, 0.5, 0.4),
        (1903, 0.6, 0.5),
    ):
        records.append(_record("recurrent", seed, recurrent))
        records.append(_record("pooled", seed, pooled))
    return records


def _drive_backup() -> dict[str, object]:
    return {
        "drive_folder": {
            "id": "folder-id",
            "url": "https://drive.google.com/drive/folders/folder-id",
        },
        "uploads": [{"experiment_id": index} for index in range(6)],
    }


def test_build_summary_uses_paired_frozen_runs() -> None:
    summary = build_summary(
        _records(),
        _drive_backup(),
        completed_at="2026-01-01T00:00:00+00:00",
    )

    assert summary["run_count"] == 6
    assert summary["total_optimizer_steps"] == 30_000
    assert summary["primary_metric_winner"] == "recurrent"
    assert summary["paired"]["mean_recurrent_minus_pooled"] == pytest.approx(
        0.1
    )
    assert summary["paired"]["recurrent_seed_wins"] == 3
    assert summary["groups"]["recurrent"]["mean"] == pytest.approx(0.5)
    assert summary["groups"]["pooled"]["mean"] == pytest.approx(0.4)

    markdown = render_markdown(summary)
    assert "Post-sealed diagnostic only" in markdown
    assert "Google Drive" in markdown
    assert "**recurrent**" in markdown


def test_record_validation_rejects_sealed_reports() -> None:
    record = deepcopy(_record("recurrent", 1701, 0.4))
    record["reports"]["test_sealed_v0_2"] = {
        "primary_autonomous_success": 1.0
    }

    with pytest.raises(RuntimeError, match="sealed report"):
        validate_record(record)


def test_record_validation_rejects_training_drift() -> None:
    record = _record("pooled", 1701, 0.4)
    record["steps"] = 4_999

    with pytest.raises(RuntimeError, match="step count drift"):
        validate_record(record)
