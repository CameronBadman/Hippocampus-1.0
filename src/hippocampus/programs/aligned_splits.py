from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Sequence

from .generator import GeneratorConfig, GraphProgramGenerator
from .schema import GraphProgramCase, ProgramFamily, TerminationDecision


V04_DATASET_VERSION = "spider-programs-v0.4-aligned-dev"
V04_1_DATASET_VERSION = "spider-programs-v0.4.1-aligned-evidence-dev"


@dataclass(frozen=True, slots=True)
class AlignedDevSplitSpec:
    """One non-sealed partition in the Spider v0.4 development campaign."""

    name: str
    case_count: int
    seed_start: int
    dataset_version: str = V04_DATASET_VERSION
    sealed: bool = False

    def __post_init__(self) -> None:
        if self.case_count <= 0:
            raise ValueError("case_count must be positive")
        if self.case_count % 128 != 0:
            raise ValueError(
                "aligned split sizes must be divisible by the 128-case "
                "family/outcome/graph/path stratum cycle"
            )
        if self.sealed:
            raise ValueError("v0.4 development specifications may not be sealed")


@dataclass(frozen=True, slots=True)
class AlignedDevSplitManifest:
    spec: AlignedDevSplitSpec
    materialized_case_count: int
    case_ids: tuple[str, ...]
    base_case_ids: tuple[str, ...]
    case_sha256: tuple[str, ...]
    distributions: dict[str, dict[str, int]]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": asdict(self.spec),
            "materialized_case_count": self.materialized_case_count,
            "case_ids": list(self.case_ids),
            "base_case_ids": list(self.base_case_ids),
            "case_sha256": list(self.case_sha256),
            "distributions": self.distributions,
            "sha256": self.sha256,
        }


def default_aligned_dev_specs() -> tuple[AlignedDevSplitSpec, ...]:
    """Return the four preregistered, disjoint, non-sealed v0.4 partitions."""

    return (
        AlignedDevSplitSpec("training", 8_192, 410_000),
        AlignedDevSplitSpec("model_selection", 512, 510_000),
        AlignedDevSplitSpec("calibration", 512, 520_000),
        AlignedDevSplitSpec("development_evaluation", 1_024, 530_000),
    )


def default_aligned_evidence_specs() -> tuple[AlignedDevSplitSpec, ...]:
    """Return the versioned evidence-only amendment to the v0.4 splits."""

    return (
        AlignedDevSplitSpec(
            "training",
            8_192,
            610_000,
            dataset_version=V04_1_DATASET_VERSION,
        ),
        AlignedDevSplitSpec(
            "model_selection",
            512,
            710_000,
            dataset_version=V04_1_DATASET_VERSION,
        ),
        AlignedDevSplitSpec(
            "calibration",
            512,
            720_000,
            dataset_version=V04_1_DATASET_VERSION,
        ),
        AlignedDevSplitSpec(
            "development_evaluation",
            1_024,
            730_000,
            dataset_version=V04_1_DATASET_VERSION,
        ),
    )


_GRAPH_SIZE_BUCKETS = (8, 16, 24, 32)
_PATH_LENGTH_BUCKETS = (1, 2, 3, 4)


def _negative_controls(index: int, family: ProgramFamily):
    outcome_group = index // len(ProgramFamily)
    negative_variant = (outcome_group // 2) % 3
    if negative_variant == 1:
        return (
            TerminationDecision.UNKNOWN_INCOMPLETE,
            family is ProgramFamily.LATEST_VALID,
        )
    if negative_variant == 2:
        return TerminationDecision.UNKNOWN_UNSUPPORTED, False
    return None, False


def generate_aligned_dev_cases(
    spec: AlignedDevSplitSpec,
    *,
    limit: int | None = None,
) -> tuple[GraphProgramCase, ...]:
    """Generate a deterministic, jointly stratified v0.4 partition.

    Each 128-case cycle crosses four program families, two outcome groups,
    four exact graph sizes, and four requested path-length buckets. Some
    family semantics constrain observed path length and evidence cardinality;
    the manifest records those observed marginals instead of pretending those
    structurally coupled variables are independent.
    """

    case_count = spec.case_count if limit is None else min(limit, spec.case_count)
    if case_count <= 0:
        raise ValueError("split case limit must be positive")
    families = tuple(ProgramFamily)
    generator_cache: dict[tuple[int, int], GraphProgramGenerator] = {}
    cases: list[GraphProgramCase] = []
    for index in range(case_count):
        family = families[index % len(families)]
        answerable = (index // len(families)) % 2 == 0
        graph_size = _GRAPH_SIZE_BUCKETS[(index // 8) % 4]
        path_length = _PATH_LENGTH_BUCKETS[(index // 32) % 4]
        cache_key = (graph_size, path_length)
        generator = generator_cache.get(cache_key)
        if generator is None:
            generator = GraphProgramGenerator(
                GeneratorConfig(
                    min_nodes=graph_size,
                    max_nodes=graph_size,
                    min_path_length=path_length,
                    max_path_length=path_length,
                    generator_version=spec.dataset_version,
                )
            )
            generator_cache[cache_key] = generator
        unknown_decision = None
        context_budget_exhausted = False
        if not answerable:
            unknown_decision, context_budget_exhausted = _negative_controls(
                index,
                family,
            )
        cases.append(
            generator.generate(
                family=family,
                seed=spec.seed_start + index,
                answerable=answerable,
                require_multiple_paths=(
                    family is ProgramFamily.REACHABILITY and index % 3 == 0
                ),
                unknown_decision=unknown_decision,
                context_budget_exhausted=context_budget_exhausted,
            )
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("v0.4 generator produced duplicate case IDs")
    return tuple(cases)


def generate_aligned_evidence_cases(
    spec: AlignedDevSplitSpec,
    *,
    limit: int | None = None,
) -> tuple[GraphProgramCase, ...]:
    """Generate the v0.4.1 evidence amendment without unsupported queries.

    Unsupported-interface recognition belongs to learned termination, which is
    explicitly deferred in v0.4. Evidence experiments instead alternate
    ordinary absent/conflict outcomes with exact budget-incomplete outcomes.
    This keeps query row cardinality matched across answerability classes.
    """

    if spec.dataset_version != V04_1_DATASET_VERSION:
        raise ValueError("evidence amendment requires the v0.4.1 dataset version")
    case_count = spec.case_count if limit is None else min(limit, spec.case_count)
    if case_count <= 0:
        raise ValueError("split case limit must be positive")
    families = tuple(ProgramFamily)
    generator_cache: dict[tuple[int, int], GraphProgramGenerator] = {}
    cases: list[GraphProgramCase] = []
    for index in range(case_count):
        family = families[index % len(families)]
        answerable = (index // len(families)) % 2 == 0
        graph_size = _GRAPH_SIZE_BUCKETS[(index // 8) % 4]
        path_length = _PATH_LENGTH_BUCKETS[(index // 32) % 4]
        cache_key = (graph_size, path_length)
        generator = generator_cache.get(cache_key)
        if generator is None:
            generator = GraphProgramGenerator(
                GeneratorConfig(
                    min_nodes=graph_size,
                    max_nodes=graph_size,
                    min_path_length=path_length,
                    max_path_length=path_length,
                    generator_version=spec.dataset_version,
                )
            )
            generator_cache[cache_key] = generator
        unknown_decision = None
        context_budget_exhausted = False
        if not answerable:
            outcome_group = index // len(families)
            if (outcome_group // 2) % 2 == 1:
                unknown_decision = TerminationDecision.UNKNOWN_INCOMPLETE
                context_budget_exhausted = (
                    family is ProgramFamily.LATEST_VALID
                )
        cases.append(
            generator.generate(
                family=family,
                seed=spec.seed_start + index,
                answerable=answerable,
                require_multiple_paths=(
                    family is ProgramFamily.REACHABILITY and index % 3 == 0
                ),
                unknown_decision=unknown_decision,
                context_budget_exhausted=context_budget_exhausted,
            )
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("v0.4.1 generator produced duplicate case IDs")
    return tuple(cases)


def _canonical_case_bytes(case: GraphProgramCase) -> bytes:
    return json.dumps(
        asdict(case),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _observed_path_length(case: GraphProgramCase) -> int:
    return max((len(path) - 1 for path in case.trace.valid_paths), default=0)


def _distribution(
    cases: Sequence[GraphProgramCase],
    key,
) -> dict[str, int]:
    return dict(sorted(Counter(str(key(case)) for case in cases).items()))


def build_aligned_dev_manifest(
    spec: AlignedDevSplitSpec,
    cases: Sequence[GraphProgramCase],
) -> AlignedDevSplitManifest:
    """Hash actual generated content and record the controlled marginals."""

    if not cases:
        raise ValueError("a split manifest requires at least one case")
    if len(cases) > spec.case_count:
        raise ValueError("materialized cases exceed the registered split size")
    case_ids = tuple(case.case_id for case in cases)
    base_case_ids = tuple(case.base_case_id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case IDs in one split must be unique")
    if len(set(base_case_ids)) != len(base_case_ids):
        raise ValueError(
            "base cases and their related views must be grouped before hashing"
        )
    fingerprints = tuple(
        hashlib.sha256(_canonical_case_bytes(case)).hexdigest()
        for case in cases
    )
    distributions = {
        "family": _distribution(cases, lambda case: case.family.value),
        "outcome": _distribution(
            cases,
            lambda case: "answerable" if case.answerable else "unknown",
        ),
        "required_evidence_cardinality": _distribution(
            cases,
            lambda case: len(case.evidence_nodes),
        ),
        "observed_path_length": _distribution(cases, _observed_path_length),
        "graph_size": _distribution(cases, lambda case: len(case.nodes)),
    }
    payload = {
        "spec": asdict(spec),
        "case_ids": case_ids,
        "base_case_ids": base_case_ids,
        "case_sha256": fingerprints,
        "distributions": distributions,
    }
    aggregate = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return AlignedDevSplitManifest(
        spec=spec,
        materialized_case_count=len(cases),
        case_ids=case_ids,
        base_case_ids=base_case_ids,
        case_sha256=fingerprints,
        distributions=distributions,
        sha256=aggregate,
    )
