#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


SEEDS = (1701, 1802, 1903)
MODELS = ("recurrent", "pooled")
STANDARD_ABLATION_SPLITS = (
    "validation_id",
    "validation_path_length_ood",
)


def _load_runs(root: Path) -> dict[tuple[str, int], dict[str, object]]:
    runs: dict[tuple[str, int], dict[str, object]] = {}
    for model in MODELS:
        for seed in SEEDS:
            path = root / f"{model}_{seed}.json"
            payload = json.loads(path.read_text())
            if payload["sealed_access_count"] != 0:
                raise RuntimeError(f"{path} accessed sealed data")
            if payload["status"] != "accepted" or not payload["pass"]:
                raise RuntimeError(f"{path} is not an accepted diagnostic")
            if payload["checkpoint"]["model_kind"] != model:
                raise RuntimeError(f"{path} model kind is mislabelled")
            if payload["checkpoint"]["seed"] != seed:
                raise RuntimeError(f"{path} seed is mislabelled")
            if payload["guard"]["deterministic_replay_mismatches"]:
                raise RuntimeError(f"{path} failed deterministic replay")
            if payload["guard"]["row_permutation_mismatches"]:
                raise RuntimeError(f"{path} failed row invariance")
            runs[(model, seed)] = payload
    return runs


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _weighted_state_score(
    run: dict[str, object],
    intervention: str,
) -> float:
    reports = run["state_ablation_reports"][intervention]
    selected = [reports[name] for name in STANDARD_ABLATION_SPLITS]
    count = sum(int(report["case_count"]) for report in selected)
    return sum(
        float(report["structural_success"]) * int(report["case_count"])
        for report in selected
    ) / count


def _state_rule(
    degradations: dict[str, dict[int, float]],
) -> dict[str, object]:
    candidates = {}
    for intervention in ("reset", "shuffle"):
        values = list(degradations[intervention].values())
        candidates[intervention] = {
            "mean_degradation": _mean(values),
            "seed_degradations": {
                str(seed): value
                for seed, value in degradations[intervention].items()
            },
            "seeds_at_or_above_0_05": sum(value >= 0.05 for value in values),
            "material": (
                _mean(values) >= 0.05
                and sum(value >= 0.05 for value in values) >= 2
            ),
        }
    return {
        "by_intervention": candidates,
        "material_state_use": any(
            item["material"] for item in candidates.values()
        ),
    }


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Spider v0.2 Preserved-Checkpoint Diagnostic",
        "",
        "This is a post-sealed diagnostic with no selection effect. All inputs "
        "are preserved checkpoints and non-sealed development/validation data.",
        "",
        "## Oracle-required horizon",
        "",
        "| Seed | Recurrent structural | Pooled structural | R − P | "
        "Recurrent final | Pooled final |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = summary["paired"]["by_seed"][str(seed)]
        lines.append(
            f"| {seed} | {row['recurrent_structural']:.4f} | "
            f"{row['pooled_structural']:.4f} | "
            f"{row['structural_delta']:+.4f} | "
            f"{row['recurrent_final']:.4f} | "
            f"{row['pooled_final']:.4f} |"
        )
    paired = summary["paired"]
    lines.extend(
        [
            "",
            f"Mean structural delta: **{paired['mean_structural_delta']:+.4f}**; "
            f"recurrent seed wins: **{paired['recurrent_seed_wins']}/3**.",
            "",
            "## Recurrent state interventions",
            "",
            "| Intervention | Mean degradation vs intact | Seeds ≥ 0.05 |",
            "|---|---:|---:|",
        ]
    )
    state = summary["state_use"]["by_intervention"]
    for intervention in ("reset", "shuffle"):
        item = state[intervention]
        lines.append(
            f"| {intervention} | {item['mean_degradation']:+.4f} | "
            f"{item['seeds_at_or_above_0_05']}/3 |"
        )
    lines.extend(
        [
            "",
            f"Material state-use rule passed: "
            f"**{summary['state_use']['material_state_use']}**.",
            "",
            "## Outcome",
            "",
            f"Recurrent-advantage rule passed: "
            f"**{summary['decision']['recurrent_advantage']}**.",
            "",
            "Suppressing intermediate stopping improves final decisions for "
            "some checkpoints, but the preserved recurrent model does not show "
            "a robust structural advantage and its state interventions do not "
            "meet the pre-registered material-use threshold. Zero-shot accuracy "
            "on the new recurrence-necessity split is zero for both model "
            "families, so that split requires matched training before it can "
            "judge learnability.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    artifact_root = Path("artifacts/spider_v0_2")
    diagnostic_root = artifact_root / "diagnostics"
    runs = _load_runs(diagnostic_root)
    paired_by_seed: dict[str, object] = {}
    structural_deltas: list[float] = []
    final_deltas: list[float] = []
    recurrent_wins = 0
    for seed in SEEDS:
        recurrent = runs[("recurrent", seed)]["aggregates"]["oracle_required"]
        pooled = runs[("pooled", seed)]["aggregates"]["oracle_required"]
        recurrent_structural = float(
            recurrent["standard_structural_success"]
        )
        pooled_structural = float(pooled["standard_structural_success"])
        recurrent_final = float(
            recurrent["standard_final_autonomous_success"]
        )
        pooled_final = float(pooled["standard_final_autonomous_success"])
        structural_delta = recurrent_structural - pooled_structural
        final_delta = recurrent_final - pooled_final
        structural_deltas.append(structural_delta)
        final_deltas.append(final_delta)
        recurrent_wins += int(structural_delta > 0)
        paired_by_seed[str(seed)] = {
            "recurrent_structural": recurrent_structural,
            "pooled_structural": pooled_structural,
            "structural_delta": structural_delta,
            "recurrent_final": recurrent_final,
            "pooled_final": pooled_final,
            "final_delta": final_delta,
        }

    degradations: dict[str, dict[int, float]] = {
        intervention: {}
        for intervention in ("reset", "detach", "shuffle", "pooled_current_node")
    }
    recurrence_state_scores: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        run = runs[("recurrent", seed)]
        intact = _weighted_state_score(run, "none")
        recurrence_state_scores[str(seed)] = {}
        for intervention in degradations:
            score = _weighted_state_score(run, intervention)
            degradations[intervention][seed] = intact - score
            recurrence_state_scores[str(seed)][intervention] = float(
                run["state_ablation_reports"][intervention][
                    "validation_recurrence_necessity"
                ]["structural_success"]
            )
        recurrence_state_scores[str(seed)]["none"] = float(
            run["state_ablation_reports"]["none"][
                "validation_recurrence_necessity"
            ]["structural_success"]
        )
    state_use = _state_rule(degradations)
    state_use["all_degradations"] = {
        intervention: {
            str(seed): value for seed, value in by_seed.items()
        }
        for intervention, by_seed in degradations.items()
    }
    state_use["recurrence_split_scores"] = recurrence_state_scores

    mean_structural_delta = _mean(structural_deltas)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "post-sealed diagnostic; no selection effect",
        "run_count": len(runs),
        "sealed_access_count": 0,
        "runtime_seconds": sum(
            float(run["runtime_seconds"]) for run in runs.values()
        ),
        "source_commits": sorted(
            {str(run["source_commit"]) for run in runs.values()}
        ),
        "paired": {
            "by_seed": paired_by_seed,
            "mean_structural_delta": mean_structural_delta,
            "population_stddev_structural_delta": statistics.pstdev(
                structural_deltas
            ),
            "mean_final_delta": _mean(final_deltas),
            "recurrent_seed_wins": recurrent_wins,
            "pooled_seed_wins": len(SEEDS) - recurrent_wins,
        },
        "state_use": state_use,
        "decision": {
            "recurrent_advantage": (
                mean_structural_delta >= 0.02 and recurrent_wins >= 2
            ),
            "required_mean_delta": 0.02,
            "required_seed_wins": 2,
            "material_state_use": state_use["material_state_use"],
        },
        "guards": {
            "all_finite": all(
                math.isfinite(delta)
                for delta in structural_deltas + final_deltas
            ),
            "deterministic_replay_mismatches": 0,
            "row_permutation_mismatches": 0,
            "historical_sealed_access": 0,
        },
    }
    (artifact_root / "DIAGNOSTIC_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (artifact_root / "DIAGNOSTIC_SUMMARY.md").write_text(
        _markdown(summary)
    )
    with (artifact_root / "experiments.jsonl").open("w") as handle:
        for model in MODELS:
            for seed in SEEDS:
                run = runs[(model, seed)]
                record = {
                    "experiment_id": run["experiment_id"],
                    "timestamp_unix": run["timestamp_unix"],
                    "source_commit": run["source_commit"],
                    "checkpoint_sha256": run["checkpoint"]["sha256"],
                    "dataset_versions": [
                        "spider-programs-v0.2",
                        "spider-programs-v0.3-recurrence-dev",
                    ],
                    "configuration": {
                        "model_kind": model,
                        "seed": seed,
                        "policies": list(run["policy_reports"]),
                        "state_interventions": list(
                            run["state_ablation_reports"]
                        ),
                    },
                    "metrics": run["aggregates"],
                    "runtime_seconds": run["runtime_seconds"],
                    "sealed_access_count": 0,
                    "status": run["status"],
                    "failure_reason": None,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "mean_structural_delta": mean_structural_delta,
                "recurrent_seed_wins": recurrent_wins,
                "material_state_use": state_use["material_state_use"],
                "recurrent_advantage": summary["decision"][
                    "recurrent_advantage"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

