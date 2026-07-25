from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SplitSpec:
    name: str
    case_count: int
    seed_start: int
    min_nodes: int
    max_nodes: int
    min_path_length: int
    max_path_length: int
    domain: str = "train"
    cardinality_scale: float = 1.0
    held_out_topology: bool = False
    held_out_composition: bool = False
    sealed: bool = False
    generator_version: str = "spider-programs-v0.1"

    def __post_init__(self) -> None:
        if self.case_count <= 0:
            raise ValueError("case_count must be positive")


@dataclass(frozen=True, slots=True)
class SplitManifest:
    spec: SplitSpec
    case_ids: tuple[str, ...]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": asdict(self.spec),
            "case_ids": list(self.case_ids),
            "sha256": self.sha256,
        }


def _scaled(value: int, scale: float) -> int:
    return max(1, round(value * scale))


def default_split_specs(*, case_scale: float = 1.0) -> tuple[SplitSpec, ...]:
    if case_scale <= 0:
        raise ValueError("case_scale must be positive")
    return (
        SplitSpec("train", _scaled(512, case_scale), 10_000, 8, 32, 1, 4),
        SplitSpec(
            "validation_id",
            _scaled(128, case_scale),
            20_000,
            8,
            32,
            1,
            4,
        ),
        SplitSpec(
            "validation_graph_size_ood",
            _scaled(96, case_scale),
            30_000,
            64,
            128,
            1,
            8,
        ),
        SplitSpec(
            "validation_path_length_ood",
            _scaled(96, case_scale),
            40_000,
            16,
            64,
            5,
            8,
        ),
        SplitSpec(
            "validation_topology_ood",
            _scaled(96, case_scale),
            50_000,
            16,
            64,
            2,
            8,
            held_out_topology=True,
        ),
        SplitSpec(
            "validation_cardinality_ood",
            _scaled(96, case_scale),
            60_000,
            8,
            32,
            1,
            4,
            cardinality_scale=3.0,
        ),
        SplitSpec(
            "validation_equivalent_view_ood",
            _scaled(96, case_scale),
            70_000,
            8,
            32,
            1,
            4,
            domain="held_out",
        ),
        SplitSpec(
            "validation_composition_ood",
            _scaled(96, case_scale),
            80_000,
            16,
            64,
            2,
            8,
            held_out_composition=True,
        ),
        SplitSpec(
            "test_sealed",
            _scaled(256, case_scale),
            90_000,
            8,
            128,
            1,
            8,
            domain="sealed",
            sealed=True,
        ),
    )


def build_split_manifest(spec: SplitSpec) -> SplitManifest:
    case_ids = tuple(
        hashlib.sha256(
            (
                f"{spec.generator_version}|{spec.name}|"
                f"{spec.seed_start + index}|{index}"
            ).encode()
        ).hexdigest()[:24]
        for index in range(spec.case_count)
    )
    payload = json.dumps(
        {
            "spec": asdict(spec),
            "case_ids": case_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SplitManifest(
        spec=spec,
        case_ids=case_ids,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
