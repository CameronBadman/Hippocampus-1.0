#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any
import zlib

from style_presets import rcparams


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/spider_v0_1"
LEDGER = ARTIFACT_ROOT / "experiments.jsonl"
RESULTS_TSV = ARTIFACT_ROOT / "autoresearch-results.tsv"
OLD_CHECKPOINT = (
    ROOT
    / "artifacts/spider_v0/autoresearch/runs/"
    "E003-recurrent-standard/checkpoint.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Spider v0.1 AutoResearch matrix."
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--train-cases", type=int, default=512)
    parser.add_argument(
        "--eval-cases",
        type=int,
        default=0,
        help="0 uses every case in each validation split",
    )
    parser.add_argument("--primary-seed", type=int, default=401)
    parser.add_argument(
        "--replicate-seeds",
        type=int,
        nargs=2,
        default=(502, 603),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--skip-guard-tests", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _definitions(args: argparse.Namespace) -> list[dict[str, Any]]:
    matrix = [
        (
            "E000-old-checkpoint-corrected",
            "configs/spider_v0_1/recurrent_oracle.json",
            "The old checkpoint diagnoses controller correction without "
            "altering historical v0 evidence.",
            True,
        ),
        (
            "E001-unified-oracle",
            "configs/spider_v0_1/recurrent_oracle.json",
            "Unified post-transition state supervision removes feature and "
            "termination-boundary mismatch.",
            False,
        ),
        (
            "E002-scheduled-closed-loop",
            "configs/spider_v0_1/recurrent_scheduled.json",
            "Independent scheduled actions expose recoverable model states "
            "and reduce autonomous one-round collapse.",
            False,
        ),
        (
            "E003-balanced-evidence",
            "configs/spider_v0_1/recurrent_evidence.json",
            "Balanced and set-level evidence pressure improves autonomous "
            "evidence recall.",
            False,
        ),
        (
            "E004-hierarchical-recurrent",
            "configs/spider_v0_1/recurrent_hierarchical.json",
            "Hierarchical stop/reason supervision improves autonomous success.",
            False,
        ),
        (
            "E005-hierarchical-pooled",
            "configs/spider_v0_1/pooled_hierarchical.json",
            "A matched pooled controller is the fixed architectural control.",
            False,
        ),
    ]
    definitions = [
        {
            "id": experiment_id,
            "config": config,
            "hypothesis": hypothesis,
            "diagnostic": diagnostic,
            "seed": args.primary_seed,
        }
        for experiment_id, config, hypothesis, diagnostic in matrix
    ]
    for seed in args.replicate_seeds:
        definitions.extend(
            (
                {
                    "id": f"F-E004-hierarchical-recurrent-s{seed}",
                    "config": "configs/spider_v0_1/recurrent_hierarchical.json",
                    "hypothesis": "Recurrent finalist independent-seed replicate.",
                    "diagnostic": False,
                    "seed": seed,
                },
                {
                    "id": f"F-E005-hierarchical-pooled-s{seed}",
                    "config": "configs/spider_v0_1/pooled_hierarchical.json",
                    "hypothesis": "Pooled finalist independent-seed replicate.",
                    "diagnostic": False,
                    "seed": seed,
                },
            )
        )
    return definitions


def _read_ledger() -> list[dict[str, Any]]:
    if not LEDGER.exists():
        return []
    return [
        json.loads(line)
        for line in LEDGER.read_text().splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_preregistration(
    args: argparse.Namespace,
    source_commit: str,
) -> None:
    split_index = _load_json(
        ARTIFACT_ROOT / "splits/MANIFEST_INDEX.json"
    )
    payload = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "dataset_version": "spider-programs-v0.2",
        "dataset_split_digest": split_index["aggregate_sha256"],
        "primary_metric": "complete-validation autonomous success",
        "threshold_calibration": (
            "maximum evidence F1 on complete validation_id, "
            "higher recall then lower threshold tie-break"
        ),
        "accepted_experiment_budget": 10,
        "training_steps": args.steps,
        "training_case_count": args.train_cases,
        "evaluation_case_limit": args.eval_cases,
        "primary_seed": args.primary_seed,
        "replicate_seeds": list(args.replicate_seeds),
        "finalist_rule": (
            "three-seed mean autonomous success; evidence F1, false-answer "
            "rate, arcs scored, experiment ID tie-breaks"
        ),
        "sealed_policy": (
            "test_sealed_v0_2 remains unopened until finalist manifest exists"
        ),
        "old_v0_sealed_accesses": 0,
    }
    path = ARTIFACT_ROOT / "PREREGISTRATION.json"
    if path.exists() and _load_json(path) != payload:
        # Timestamp is intentionally ignored when checking a resumed run.
        existing = _load_json(path)
        comparable = {key: value for key, value in payload.items() if key != "frozen_at"}
        existing_comparable = {
            key: value for key, value in existing.items() if key != "frozen_at"
        }
        if existing_comparable != comparable:
            raise RuntimeError("existing v0.1 preregistration differs from this run")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _controlled_config(
    definition: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = deepcopy(_load_json(ROOT / definition["config"]))
    config["name"] = definition["id"]
    config["training"]["seed"] = definition["seed"]
    config["training"]["steps"] = args.steps
    config["training"]["device"] = args.device
    config["training"]["dtype"] = args.dtype
    return config


def _run_guard() -> None:
    subprocess.run(
        [str(ROOT / ".venv/bin/pytest"), "-q"],
        cwd=ROOT,
        check=True,
        timeout=300,
    )


def _failure_record(
    definition: dict[str, Any],
    *,
    attempt: int,
    source_commit: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "pass": False,
        "status": "crashed",
        "experiment_id": definition["id"],
        "attempt": attempt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "hypothesis": definition["hypothesis"],
        "score": None,
        "failure_reason": reason,
        "sealed_access_count": 0,
    }


def _run_definition(
    definition: dict[str, Any],
    args: argparse.Namespace,
    *,
    source_commit: str,
    attempt: int,
) -> dict[str, Any]:
    run_id = definition["id"]
    config_dir = ARTIFACT_ROOT / "autoresearch/configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{run_id}.json"
    config_path.write_text(
        json.dumps(
            _controlled_config(definition, args),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    output_dir = ARTIFACT_ROOT / "autoresearch/runs" / run_id
    command = [
        "timeout",
        "5m",
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/spider_v0_1_evaluator.py"),
        "--config",
        str(config_path),
        "--experiment-id",
        run_id,
        "--output-dir",
        str(output_dir),
        "--train-cases",
        str(args.train_cases),
        "--eval-cases",
        str(args.eval_cases),
        "--steps",
        str(args.steps),
        "--seed",
        str(definition["seed"]),
    ]
    if definition["diagnostic"]:
        command.extend(("--diagnostic-checkpoint", str(OLD_CHECKPOINT)))
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=310,
        )
    except subprocess.TimeoutExpired:
        return _failure_record(
            definition,
            attempt=attempt,
            source_commit=source_commit,
            reason="TIMEOUT: experiment exceeded five-minute budget",
        )
    if completed.returncode != 0:
        reason = (
            completed.stderr.strip()[-4000:]
            or completed.stdout.strip()[-4000:]
            or f"exit code {completed.returncode}"
        )
        return _failure_record(
            definition,
            attempt=attempt,
            source_commit=source_commit,
            reason=reason,
        )
    metrics_path = output_dir / "metrics.json"
    try:
        result = _load_json(metrics_path)
    except (OSError, json.JSONDecodeError) as exc:
        return _failure_record(
            definition,
            attempt=attempt,
            source_commit=source_commit,
            reason=f"invalid evaluator metrics artifact: {exc}",
        )
    result["attempt"] = attempt
    result["hypothesis"] = definition["hypothesis"]
    if result.get("source_commit") != source_commit:
        result["pass"] = False
        result["status"] = "guard_violation"
        result["failure_reason"] = "source commit changed during matrix"
    return result


def _write_summary(records: list[dict[str, Any]]) -> None:
    accepted = [
        record for record in records if record.get("status") == "accepted"
    ]
    lines = [
        "# Spider v0.1 AutoResearch summary",
        "",
        "| Experiment | Seed | Autonomous success | Evidence F1 | "
        "One-round stop | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        reports = record.get("reports", {})
        validation = (
            reports.get("validation_id", {})
            if isinstance(reports, dict)
            else {}
        )
        evidence = validation.get("evidence", {})
        rollout = validation.get("rollout", {})
        score = record.get("score")
        lines.append(
            "| {id} | {seed} | {score} | {evidence} | {one_round} | "
            "{status} |".format(
                id=record.get("experiment_id", "?"),
                seed=record.get("seed", "?"),
                score=(
                    f"{float(score):.4f}" if score is not None else "—"
                ),
                evidence=(
                    f"{float(evidence.get('f1', 0.0)):.4f}"
                    if evidence
                    else "—"
                ),
                one_round=(
                    f"{float(rollout.get('one_round_stop_rate', 0.0)):.4f}"
                    if rollout
                    else "—"
                ),
                status=record.get("status", "?"),
            )
        )
    lines.extend(
        (
            "",
            f"Accepted records: {len(accepted)} / 10.",
            "The Spider v0 sealed result was not accessed.",
            "",
        )
    )
    (ARTIFACT_ROOT / "EXPERIMENT_SUMMARY.md").write_text(
        "\n".join(lines)
    )


def _write_tsv(records: list[dict[str, Any]]) -> None:
    lines = [
        "iteration\texperiment_id\thypothesis\tscore\tbest_so_far\tstatus"
        "\truntime_seconds\tnotes"
    ]
    best = float("-inf")
    for index, record in enumerate(records, start=1):
        score = record.get("score")
        if score is not None:
            best = max(best, float(score))
        lines.append(
            "\t".join(
                (
                    str(index),
                    str(record.get("experiment_id", "")),
                    str(record.get("hypothesis", "")).replace("\t", " "),
                    "" if score is None else str(score),
                    "" if best == float("-inf") else str(best),
                    str(record.get("status", "")),
                    str(record.get("runtime_seconds", "")),
                    str(record.get("failure_reason") or "").replace(
                        "\t", " "
                    ),
                )
            )
        )
    RESULTS_TSV.write_text("\n".join(lines) + "\n")


def _set_pixel(
    image: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    if 0 <= x < width and 0 <= y < height:
        offset = 3 * (y * width + x)
        image[offset : offset + 3] = bytes(color)


def _line(
    image: bytearray,
    width: int,
    height: int,
    first: tuple[int, int],
    second: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x0, y0 = first
    x1, y1 = second
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _set_pixel(image, width, height, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += step_x
        if doubled <= dx:
            error += dx
            y0 += step_y


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_progress(records: list[dict[str, Any]]) -> None:
    style = rcparams()
    width = int(style["width"])
    height = int(style["height"])
    margin = int(style["margin"])
    background = style["background"]
    image = bytearray(bytes(background) * (width * height))
    foreground = style["foreground"]
    grid = style["grid"]
    _line(
        image,
        width,
        height,
        (margin, height - margin),
        (width - margin, height - margin),
        foreground,
    )
    _line(
        image,
        width,
        height,
        (margin, margin),
        (margin, height - margin),
        foreground,
    )
    for fraction in (0.25, 0.5, 0.75, 1.0):
        y = round(height - margin - fraction * (height - 2 * margin))
        _line(
            image,
            width,
            height,
            (margin, y),
            (width - margin, y),
            grid,
        )
    scored = [
        (index, float(record["score"]), record)
        for index, record in enumerate(records, start=1)
        if record.get("score") is not None
    ]
    points: list[tuple[int, int]] = []
    best_points: list[tuple[int, int]] = []
    best = 0.0
    denominator = max(1, len(records) - 1)
    for index, score, record in scored:
        x = margin + round(
            (index - 1) / denominator * (width - 2 * margin)
        )
        y = height - margin - round(
            max(0.0, min(1.0, score)) * (height - 2 * margin)
        )
        points.append((x, y))
        best = max(best, score)
        best_y = height - margin - round(
            max(0.0, min(1.0, best)) * (height - 2 * margin)
        )
        best_points.append((x, best_y))
        color = (
            style["diagnostic"]
            if str(record.get("experiment_id", "")).startswith("E000")
            else style["kept"]
            if record.get("status") == "accepted"
            else style["guard_violation"]
        )
        for offset_x in range(-4, 5):
            for offset_y in range(-4, 5):
                if offset_x * offset_x + offset_y * offset_y <= 16:
                    _set_pixel(
                        image,
                        width,
                        height,
                        x + offset_x,
                        y + offset_y,
                        color,
                    )
    for first, second in zip(points, points[1:], strict=False):
        _line(image, width, height, first, second, style["kept"])
    for first, second in zip(best_points, best_points[1:], strict=False):
        _line(image, width, height, first, second, style["best"])

    rows = b"".join(
        b"\x00" + bytes(image[y * width * 3 : (y + 1) * width * 3])
        for y in range(height)
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(rows, level=9))
        + _png_chunk(b"IEND", b"")
    )
    (ARTIFACT_ROOT / "progress.png").write_bytes(png)


def _write_research_log(records: list[dict[str, Any]]) -> None:
    lines = ["# Spider v0.1 research log", ""]
    for index, record in enumerate(records, start=1):
        lines.extend(
            (
                f"## Iteration {index}: {record.get('experiment_id')}",
                "",
                f"- Hypothesis: {record.get('hypothesis')}",
                f"- Status: {record.get('status')}",
                f"- Autonomous score: {record.get('score')}",
                f"- Runtime seconds: {record.get('runtime_seconds')}",
                f"- Failure: {record.get('failure_reason')}",
                "",
            )
        )
    (ARTIFACT_ROOT / "research_log.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if _git("status", "--porcelain"):
        raise RuntimeError(
            "AutoResearch requires a clean committed source tree"
        )
    source_commit = _git("rev-parse", "HEAD")
    if not OLD_CHECKPOINT.exists():
        raise FileNotFoundError(
            "E000 historical checkpoint is unavailable locally"
        )
    _write_preregistration(args, source_commit)
    if not args.skip_guard_tests:
        _run_guard()

    records = _read_ledger()
    accepted_ids = {
        str(record["experiment_id"])
        for record in records
        if record.get("status") == "accepted"
    }
    definitions = _definitions(args)
    for definition in definitions:
        if definition["id"] in accepted_ids:
            continue
        accepted = False
        for attempt in (1, 2):
            record = _run_definition(
                definition,
                args,
                source_commit=source_commit,
                attempt=attempt,
            )
            _append_jsonl(LEDGER, record)
            records.append(record)
            _write_summary(records)
            _write_tsv(records)
            _write_progress(records)
            _write_research_log(records)
            if record.get("status") == "accepted":
                accepted_ids.add(definition["id"])
                accepted = True
                break
        if not accepted:
            continue

    if not args.skip_guard_tests:
        _run_guard()
    accepted_records = [
        record for record in records if record.get("status") == "accepted"
    ]
    if len({record["experiment_id"] for record in accepted_records}) != 10:
        raise RuntimeError("fixed matrix ended without ten accepted records")
    print(
        json.dumps(
            {
                "status": "completed",
                "accepted": 10,
                "source_commit": source_commit,
                "sealed_access_count": 0,
                "ledger": str(LEDGER),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
