from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .equivalent_views import make_equivalent_view
from .generator import GeneratorConfig, GraphProgramGenerator
from .schema import GraphProgramCase, ProgramFamily, TerminationDecision


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


def default_split_specs_v0_2(
    *,
    case_scale: float = 1.0,
) -> tuple[SplitSpec, ...]:
    """Disjoint closed-loop follow-up splits.

    The v0.1 specifications above are intentionally unchanged because their
    hashes are historical evidence. v0.2 uses a separate seed namespace and
    explicit generator version.
    """

    if case_scale <= 0:
        raise ValueError("case_scale must be positive")
    version = "spider-programs-v0.2"
    return (
        SplitSpec(
            "train",
            _scaled(512, case_scale),
            210_000,
            8,
            32,
            1,
            4,
            generator_version=version,
        ),
        SplitSpec(
            "validation_id",
            _scaled(128, case_scale),
            220_000,
            8,
            32,
            1,
            4,
            generator_version=version,
        ),
        SplitSpec(
            "validation_graph_size_ood",
            _scaled(96, case_scale),
            230_000,
            64,
            128,
            1,
            8,
            generator_version=version,
        ),
        SplitSpec(
            "validation_path_length_ood",
            _scaled(96, case_scale),
            240_000,
            16,
            64,
            5,
            8,
            generator_version=version,
        ),
        SplitSpec(
            "validation_topology_ood",
            _scaled(96, case_scale),
            250_000,
            16,
            64,
            2,
            8,
            held_out_topology=True,
            generator_version=version,
        ),
        SplitSpec(
            "validation_cardinality_ood",
            _scaled(96, case_scale),
            260_000,
            8,
            32,
            1,
            4,
            cardinality_scale=3.0,
            generator_version=version,
        ),
        SplitSpec(
            "validation_equivalent_view_ood",
            _scaled(96, case_scale),
            270_000,
            8,
            32,
            1,
            4,
            domain="held_out_v0_2",
            generator_version=version,
        ),
        SplitSpec(
            "validation_composition_ood",
            _scaled(96, case_scale),
            280_000,
            16,
            64,
            2,
            8,
            held_out_composition=True,
            generator_version=version,
        ),
        SplitSpec(
            "development_rollout_stress",
            _scaled(128, case_scale),
            285_000,
            8,
            32,
            1,
            4,
            domain="development_v0_2",
            held_out_topology=True,
            generator_version=version,
        ),
        SplitSpec(
            "test_sealed_v0_2",
            _scaled(256, case_scale),
            290_000,
            8,
            128,
            1,
            8,
            domain="sealed_v0_2",
            sealed=True,
            generator_version=version,
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


def generate_split_cases(
    spec: SplitSpec,
    *,
    limit: int | None = None,
) -> tuple[GraphProgramCase, ...]:
    """Materialise a deterministic split without storing a parallel graph form."""

    case_count = spec.case_count if limit is None else min(limit, spec.case_count)
    if case_count <= 0:
        raise ValueError("split case limit must be positive")
    row_scale = max(1, round(spec.cardinality_scale))
    generator = GraphProgramGenerator(
        GeneratorConfig(
            min_nodes=spec.min_nodes,
            max_nodes=spec.max_nodes,
            min_path_length=spec.min_path_length,
            max_path_length=spec.max_path_length,
            min_summary_rows=1,
            max_summary_rows=5 * row_scale,
            min_context_rows=0,
            max_context_rows=6 * row_scale,
            max_distractor_edges=(
                16 if spec.held_out_topology else 8
            ),
            generator_version=spec.generator_version,
        )
    )
    families = tuple(ProgramFamily)
    cases: list[GraphProgramCase] = []
    for index in range(case_count):
        family = families[index % len(families)]
        outcome_group = index // len(families)
        answerable = outcome_group % 2 == 0
        unknown_decision = None
        context_budget_exhausted = False
        if not answerable:
            negative_variant = (outcome_group // 2) % 3
            if negative_variant == 1:
                unknown_decision = TerminationDecision.UNKNOWN_INCOMPLETE
                context_budget_exhausted = family is ProgramFamily.LATEST_VALID
            elif negative_variant == 2:
                unknown_decision = TerminationDecision.UNKNOWN_UNSUPPORTED
        case = generator.generate(
            family=family,
            seed=spec.seed_start + index,
            answerable=answerable,
            require_multiple_paths=(
                family is ProgramFamily.REACHABILITY
                and (
                    spec.held_out_topology
                    or spec.held_out_composition
                    or index % 3 == 0
                )
            ),
            unknown_decision=unknown_decision,
            context_budget_exhausted=context_budget_exhausted,
        )
        if spec.domain != "train":
            case = make_equivalent_view(
                case,
                seed=spec.seed_start * 17 + index,
            )
        cases.append(case)
    return tuple(cases)
