#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from hippocampus import GraphSchema, PackConfig
from hippocampus.programs import (
    RECURRENCE_DATASET_VERSION,
    SyntheticManifoldRenderer,
    default_recurrence_necessity_specs,
    default_split_specs_v0_2,
    generate_recurrence_necessity_cases,
    generate_split_cases,
    pack_rendered_cases,
)
from hippocampus.spider import (
    ControllerExecutionPolicy,
    PathStateIntervention,
    PooledScorer,
    SparseControllerConfig,
    SpiderModel,
    SpiderModelConfig,
    evaluate_closed_loop_batches,
)


STANDARD_DATASET_VERSION = "spider-programs-v0.2"
STANDARD_SPLITS = (
    "validation_id",
    "validation_graph_size_ood",
    "validation_path_length_ood",
    "validation_topology_ood",
    "validation_cardinality_ood",
    "validation_equivalent_view_ood",
    "validation_composition_ood",
)
STATE_ABLATION_SPLITS = (
    "validation_id",
    "validation_path_length_ood",
    "validation_recurrence_necessity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a preserved checkpoint without intermediate stopping."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-kind",
        choices=("recurrent", "pooled"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--eval-cases",
        type=int,
        default=0,
        help="Development smoke limit; zero evaluates complete splits.",
    )
    parser.add_argument(
        "--policies",
        default="learned,oracle_required,fixed_4,fixed_6,fixed_8",
    )
    parser.add_argument(
        "--skip-state-ablations",
        action="store_true",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(value)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostic requested but no GPU is visible")
    return resolved


def _policy(name: str, *, seed: int) -> ControllerExecutionPolicy:
    if name == "learned":
        return ControllerExecutionPolicy.learned(seed=seed)
    if name == "oracle_required":
        return ControllerExecutionPolicy.oracle_required(seed=seed)
    if name.startswith("fixed_"):
        return ControllerExecutionPolicy.fixed(
            int(name.removeprefix("fixed_")),
            seed=seed,
        )
    raise ValueError(f"unknown execution policy {name!r}")


def _pack_cases(
    cases,
    *,
    renderer: SyntheticManifoldRenderer,
    schema: GraphSchema,
    pack_config: PackConfig,
    row_seed_offset: int,
):
    return tuple(
        pack_rendered_cases(
            (case,),
            (
                renderer.render(
                    case,
                    row_permutation_seed=row_seed_offset + index,
                ),
            ),
            schema=schema,
            pack_config=pack_config,
        )
        for index, case in enumerate(cases)
    )


def _summary(report) -> dict[str, object]:
    return {
        "split": report.split,
        "case_count": report.case_count,
        "execution": report.execution,
        "structural_success": report.fixed_horizon_structural_success,
        "final_autonomous_success": report.primary_autonomous_success,
        "evidence_exact_set_accuracy": report.evidence["exact_set_accuracy"],
        "evidence_precision": report.evidence["precision"],
        "evidence_recall": report.evidence["recall"],
        "evidence_f1": report.evidence["f1"],
        "exact_valid_path_rate": report.rollout["exact_valid_path_rate"],
        "trace_validity": report.rollout["trace_validity"],
        "termination_accuracy": report.rollout["termination_accuracy"],
        "false_answer_rate": report.rollout["false_answer_rate"],
        "semantic_invalid_expansion_rate": report.rollout[
            "semantic_invalid_expansion_rate"
        ],
        "mean_rounds": report.efficiency["mean_rounds"],
        "mean_arcs_scored": report.efficiency["mean_arcs_scored"],
        "mean_contexts_read": report.efficiency["mean_contexts_read"],
        "replay_mismatches": report.invariance[
            "deterministic_replay_mismatches"
        ],
        "row_permutation_mismatches": report.invariance[
            "row_permutation_decision_mismatches"
        ],
        "maximum_score_delta": report.invariance["maximum_score_delta"],
    }


def _weighted_mean(
    reports: dict[str, dict[str, object]],
    metric: str,
) -> float:
    total = sum(int(report["case_count"]) for report in reports.values())
    return sum(
        float(report[metric]) * int(report["case_count"])
        for report in reports.values()
    ) / max(1, total)


def main() -> None:
    args = parse_args()
    if args.eval_cases < 0:
        raise ValueError("eval-cases must be non-negative")
    if args.eval_cases and args.eval_cases % 2:
        raise ValueError("matched recurrence diagnostics require an even limit")
    if "sealed" in str(args.checkpoint).lower():
        raise ValueError("sealed artifacts are forbidden diagnostic inputs")
    started = time.perf_counter()
    device = _device(args.device)
    payload = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model_config = SpiderModelConfig(**payload["model_config"])
    controller_config = SparseControllerConfig(**payload["controller_config"])
    model_type = SpiderModel if args.model_kind == "recurrent" else PooledScorer
    model = model_type(model_config).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    schema = GraphSchema(
        summary_dim=model_config.summary_dim,
        context_dim=model_config.context_dim,
        edge_dim=model_config.edge_dim,
    )
    pack_config = PackConfig(device=device, value_dtype=torch.float32)
    renderer = SyntheticManifoldRenderer(
        schema,
        query_dim=model_config.query_dim,
        seed=91_337,
    )
    manifest_path = args.checkpoint.with_name("checkpoint.manifest.json")
    checkpoint_manifest = json.loads(manifest_path.read_text())
    checkpoint_sha = _sha256(args.checkpoint)
    if checkpoint_sha != checkpoint_manifest["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash disagrees with its preserved manifest")
    evidence_threshold = float(checkpoint_manifest["evidence_threshold"])

    cases_by_split: dict[str, tuple] = {}
    standard_specs = {
        spec.name: spec
        for spec in default_split_specs_v0_2()
        if not spec.sealed and spec.name in STANDARD_SPLITS
    }
    if set(standard_specs) != set(STANDARD_SPLITS):
        raise RuntimeError("expected non-sealed v0.2 validation splits are missing")
    for name in STANDARD_SPLITS:
        cases_by_split[name] = generate_split_cases(
            standard_specs[name],
            limit=args.eval_cases or None,
        )
    recurrence_spec = next(
        spec
        for spec in default_recurrence_necessity_specs()
        if spec.name == "validation_recurrence_necessity"
    )
    cases_by_split["validation_recurrence_necessity"] = (
        generate_recurrence_necessity_cases(
            recurrence_spec,
            limit=args.eval_cases or None,
        )
    )

    batches_by_split: dict[str, tuple] = {}
    permuted_by_split: dict[str, tuple] = {}
    for split_index, (name, cases) in enumerate(cases_by_split.items()):
        batches_by_split[name] = _pack_cases(
            cases,
            renderer=renderer,
            schema=schema,
            pack_config=pack_config,
            row_seed_offset=100_000 * (split_index + 1),
        )
        permuted_by_split[name] = _pack_cases(
            cases,
            renderer=renderer,
            schema=schema,
            pack_config=pack_config,
            row_seed_offset=1_000_000 + 100_000 * (split_index + 1),
        )

    requested_policies = tuple(
        item.strip() for item in args.policies.split(",") if item.strip()
    )
    policy_reports: dict[str, dict[str, dict[str, object]]] = {}
    for policy_name in requested_policies:
        execution_policy = _policy(policy_name, seed=args.seed)
        split_reports: dict[str, dict[str, object]] = {}
        for split_name, batches in batches_by_split.items():
            report = evaluate_closed_loop_batches(
                model,
                batches,
                split=split_name,
                controller_config=controller_config,
                dataset_version=(
                    RECURRENCE_DATASET_VERSION
                    if split_name == "validation_recurrence_necessity"
                    else STANDARD_DATASET_VERSION
                ),
                evidence_threshold=evidence_threshold,
                permuted_batches=permuted_by_split[split_name],
                invariance_sample_limit=min(8, len(batches)),
                execution_policy=execution_policy,
                include_teacher_forced=False,
            )
            split_reports[split_name] = _summary(report)
        policy_reports[policy_name] = split_reports

    state_ablation_reports: dict[str, dict[str, dict[str, object]]] = {}
    if args.model_kind == "recurrent" and not args.skip_state_ablations:
        for intervention in PathStateIntervention:
            execution_policy = ControllerExecutionPolicy.oracle_required(
                intervention=intervention,
                seed=args.seed,
            )
            split_reports = {}
            for split_name in STATE_ABLATION_SPLITS:
                batches = batches_by_split[split_name]
                report = evaluate_closed_loop_batches(
                    model,
                    batches,
                    split=split_name,
                    controller_config=controller_config,
                    dataset_version=(
                        RECURRENCE_DATASET_VERSION
                        if split_name == "validation_recurrence_necessity"
                        else STANDARD_DATASET_VERSION
                    ),
                    evidence_threshold=evidence_threshold,
                    permuted_batches=permuted_by_split[split_name],
                    invariance_sample_limit=min(8, len(batches)),
                    execution_policy=execution_policy,
                    include_teacher_forced=False,
                )
                split_reports[split_name] = _summary(report)
            state_ablation_reports[intervention.value] = split_reports

    aggregates = {
        policy_name: {
            "standard_structural_success": _weighted_mean(
                {
                    name: report
                    for name, report in reports.items()
                    if name in STANDARD_SPLITS
                },
                "structural_success",
            ),
            "standard_final_autonomous_success": _weighted_mean(
                {
                    name: report
                    for name, report in reports.items()
                    if name in STANDARD_SPLITS
                },
                "final_autonomous_success",
            ),
            "recurrence_structural_success": reports[
                "validation_recurrence_necessity"
            ]["structural_success"],
            "recurrence_final_autonomous_success": reports[
                "validation_recurrence_necessity"
            ]["final_autonomous_success"],
        }
        for policy_name, reports in policy_reports.items()
    }
    all_summaries = [
        report
        for reports in policy_reports.values()
        for report in reports.values()
    ] + [
        report
        for reports in state_ablation_reports.values()
        for report in reports.values()
    ]
    finite = all(
        math.isfinite(float(report[key]))
        for report in all_summaries
        for key in (
            "structural_success",
            "final_autonomous_success",
            "evidence_f1",
        )
    )
    replay_mismatches = sum(
        int(report["replay_mismatches"]) for report in all_summaries
    )
    row_mismatches = sum(
        int(report["row_permutation_mismatches"])
        for report in all_summaries
    )
    result = {
        "experiment_id": args.experiment_id,
        "status": "accepted" if finite and not replay_mismatches and not row_mismatches else "guard_violation",
        "pass": finite and not replay_mismatches and not row_mismatches,
        "timestamp_unix": time.time(),
        "runtime_seconds": time.perf_counter() - started,
        "source_commit": _source_commit(),
        "historical_effect": "post-sealed diagnostic; no selection effect",
        "sealed_access_count": 0,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "source_commit": checkpoint_manifest["source_commit"],
            "model_kind": args.model_kind,
            "seed": args.seed,
            "model_config": asdict(model_config),
            "controller_config": asdict(controller_config),
            "evidence_threshold": evidence_threshold,
        },
        "eval_case_limit": args.eval_cases or None,
        "policy_reports": policy_reports,
        "state_ablation_reports": state_ablation_reports,
        "aggregates": aggregates,
        "guard": {
            "finite": finite,
            "deterministic_replay_mismatches": replay_mismatches,
            "row_permutation_mismatches": row_mismatches,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "status": result["status"],
                "aggregates": aggregates,
                "runtime_seconds": result["runtime_seconds"],
            },
            sort_keys=True,
        )
    )
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

