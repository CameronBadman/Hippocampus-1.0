#!/usr/bin/env python3
"""Perform the single hash-bound Spider v0.1 sealed evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from hippocampus.programs import (
    SyntheticManifoldRenderer,
    build_split_manifest,
    default_split_specs_v0_2,
    generate_split_cases,
    pack_rendered_cases,
)
from hippocampus.spider import (
    SEALED_SPLIT_V0_2,
    authorize_v0_2_sealed_evaluation,
    build_model,
    evaluate_closed_loop_batches,
    load_experiment,
    parameter_count,
    validate_v0_1_split_access,
)


DEFAULT_FINALIST = Path("artifacts/spider_v0_1/FINALIST_MANIFEST.json")
DEFAULT_SPLIT_INDEX = Path(
    "artifacts/spider_v0_1/splits/MANIFEST_INDEX.json"
)
DEFAULT_ACCESS_MARKER = Path("artifacts/spider_v0_1/SEALED_ACCESS.json")
DEFAULT_OUTPUT = Path("artifacts/spider_v0_1/SEALED_EVALUATION.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open spider-programs-v0.2 sealed data exactly once using the "
            "pre-frozen Spider v0.1 finalist."
        )
    )
    parser.add_argument("--finalist-manifest", type=Path, default=DEFAULT_FINALIST)
    parser.add_argument("--split-index", type=Path, default=DEFAULT_SPLIT_INDEX)
    parser.add_argument("--access-marker", type=Path, default=DEFAULT_ACCESS_MARKER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-v0-2-sealed",
        action="store_true",
        help="Explicitly authorize the irreversible one-time sealed opening.",
    )
    return parser.parse_args()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pack_cases(
    experiment,
    renderer: SyntheticManifoldRenderer,
    cases,
    *,
    row_seed_offset: int,
):
    return tuple(
        pack_rendered_cases(
            (case,),
            (
                renderer.render(
                    case,
                    row_permutation_seed=(
                        experiment.training_config.seed
                        + row_seed_offset
                        + index
                    ),
                ),
            ),
            schema=experiment.schema,
            pack_config=experiment.pack_config,
        )
        for index, case in enumerate(cases)
    )


def _write_access_marker(
    path: Path,
    *,
    finalist_manifest_sha256: str,
    sealed_split_sha256: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "spider-programs-v0.2",
        "split": SEALED_SPLIT_V0_2,
        "finalist_manifest_sha256": finalist_manifest_sha256,
        "sealed_split_sha256": sealed_split_sha256,
        "purpose": "one-time frozen-finalist evaluation",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def main() -> None:
    args = parse_args()
    repository_root = Path.cwd().resolve()
    authorization = authorize_v0_2_sealed_evaluation(
        args.finalist_manifest,
        args.split_index,
        access_marker_path=args.access_marker,
        output_path=args.output,
        allow_sealed=args.allow_v0_2_sealed,
        repository_root=repository_root,
    )
    experiment = load_experiment(authorization.config_path)
    if experiment.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("frozen CUDA finalist requires a visible GPU")

    model = build_model(experiment)
    checkpoint = torch.load(
        authorization.checkpoint_path,
        map_location=experiment.device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)

    specs = {
        spec.name: spec
        for spec in default_split_specs_v0_2()
    }
    sealed_spec = specs[SEALED_SPLIT_V0_2]
    validate_v0_1_split_access(sealed_spec, allow_sealed=True)
    if build_split_manifest(sealed_spec).sha256 != (
        authorization.sealed_split_sha256
    ):
        raise RuntimeError("sealed split changed after authorization")

    # This exclusive marker is written before the first case is generated.
    # A failed evaluation remains an opened sealed set and cannot be retried.
    access = _write_access_marker(
        args.access_marker,
        finalist_manifest_sha256=authorization.finalist_manifest_sha256,
        sealed_split_sha256=authorization.sealed_split_sha256,
    )
    started = time.perf_counter()
    cases = generate_split_cases(sealed_spec)
    renderer = SyntheticManifoldRenderer(
        experiment.schema,
        query_dim=experiment.query_dim,
        seed=authorization.renderer_seed,
    )
    batches = _pack_cases(
        experiment,
        renderer,
        cases,
        row_seed_offset=authorization.row_seed_offset,
    )
    permuted_batches = _pack_cases(
        experiment,
        renderer,
        cases,
        row_seed_offset=authorization.permuted_row_seed_offset,
    )
    report = evaluate_closed_loop_batches(
        model,
        batches,
        split=SEALED_SPLIT_V0_2,
        controller_config=experiment.controller_config,
        dataset_version="spider-programs-v0.2",
        evidence_threshold=authorization.evidence_threshold,
        permuted_batches=permuted_batches,
        invariance_sample_limit=authorization.invariance_sample_limit,
    )
    payload = {
        "access": access,
        "calibration_performed": False,
        "checkpoint_path": str(
            authorization.checkpoint_path.relative_to(repository_root)
        ),
        "checkpoint_sha256": _sha256(authorization.checkpoint_path),
        "config_path": str(
            authorization.config_path.relative_to(repository_root)
        ),
        "dataset_split_digest": authorization.dataset_split_digest,
        "evidence_threshold": authorization.evidence_threshold,
        "evidence_threshold_source": "validation_id",
        "experiment_id": authorization.experiment_id,
        "finalist_manifest_sha256": authorization.finalist_manifest_sha256,
        "parameter_count": parameter_count(model),
        "report": report.as_dict(),
        "runtime_seconds": time.perf_counter() - started,
        "sealed_test_opened": True,
        "source_commit_at_evaluation": _source_commit(),
        "torch": {
            "cuda_device": (
                torch.cuda.get_device_name(experiment.device)
                if experiment.device.type == "cuda"
                else None
            ),
            "cuda_runtime": torch.version.cuda,
            "device": str(experiment.device),
            "dtype": str(experiment.dtype),
            "version": torch.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "case_count": report.case_count,
                "evidence_f1": report.evidence["f1"],
                "primary_autonomous_success": (
                    report.primary_autonomous_success
                ),
                "row_permutation_mismatches": report.invariance[
                    "row_permutation_decision_mismatches"
                ],
                "sealed_test_opened": True,
                "termination_accuracy": report.rollout[
                    "termination_accuracy"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
