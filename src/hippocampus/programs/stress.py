from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum

from .schema import GraphProgramCase
from .splits import SplitSpec, generate_split_cases


class RolloutStressKind(str, Enum):
    RECOVERABLE_OFF_ORACLE = "recoverable_off_oracle"
    PARTIAL_EVIDENCE = "partial_evidence"
    FALSE_POSITIVE_CONTEXT = "false_positive_context"
    MISSED_EVIDENCE_RECOVERY = "missed_evidence_recovery"
    PREMATURE_STOP = "premature_stop"
    BUDGET_BOUNDARY = "budget_boundary"
    DUPLICATE_CONVERGING = "duplicate_converging"


@dataclass(frozen=True, slots=True)
class RolloutStressExample:
    """Supervisor-only controller state paired with one v0.2 case."""

    case: GraphProgramCase
    kind: RolloutStressKind
    frontier_nodes: tuple[int, ...]
    accumulated_evidence_nodes: tuple[int, ...]
    contexts_read_nodes: tuple[int, ...]
    round_index: int
    arcs_scored: int
    contexts_read: int
    expected_recoverable: bool

    @property
    def descriptor(self) -> dict[str, object]:
        return {
            "case_id": self.case.case_id,
            "kind": self.kind.value,
            "frontier_nodes": list(self.frontier_nodes),
            "accumulated_evidence_nodes": list(
                self.accumulated_evidence_nodes
            ),
            "contexts_read_nodes": list(self.contexts_read_nodes),
            "round_index": self.round_index,
            "arcs_scored": self.arcs_scored,
            "contexts_read": self.contexts_read,
            "expected_recoverable": self.expected_recoverable,
        }


@dataclass(frozen=True, slots=True)
class RolloutStressManifest:
    spec: SplitSpec
    descriptors: tuple[dict[str, object], ...]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": asdict(self.spec),
            "descriptors": list(self.descriptors),
            "sha256": self.sha256,
        }


def _non_oracle_node(case: GraphProgramCase, protected: set[int]) -> int:
    return next(
        (node for node in range(len(case.nodes)) if node not in protected),
        case.start_nodes[0],
    )


def _descriptor_for(
    case: GraphProgramCase,
    kind: RolloutStressKind,
) -> RolloutStressExample:
    initial = case.start_nodes
    all_oracle_frontiers = {
        node for round_ in case.trace.rounds for node in round_.frontier_nodes
    }
    evidence = case.evidence_nodes
    context_targets = tuple(
        dict.fromkeys(
            candidate.destination_node
            for round_ in case.trace.rounds
            for candidate in round_.candidates
            if candidate.context_has_value
        )
    )
    false_context = _non_oracle_node(
        case,
        set(context_targets) | set(evidence),
    )
    if kind is RolloutStressKind.RECOVERABLE_OFF_ORACLE:
        valid = (
            case.trace.rounds[1].frontier_nodes[:1]
            if len(case.trace.rounds) > 1
            else initial[:1]
        )
        distractor = _non_oracle_node(
            case,
            all_oracle_frontiers | set(evidence),
        )
        return RolloutStressExample(
            case,
            kind,
            (*valid, distractor),
            (),
            (),
            min(1, len(case.trace.rounds) - 1),
            1,
            0,
            bool(evidence),
        )
    if kind is RolloutStressKind.PARTIAL_EVIDENCE:
        partial = evidence[:1] if len(evidence) > 1 else ()
        return RolloutStressExample(
            case,
            kind,
            initial,
            partial,
            (),
            0,
            0,
            0,
            bool(set(evidence) - set(partial)),
        )
    if kind is RolloutStressKind.FALSE_POSITIVE_CONTEXT:
        return RolloutStressExample(
            case,
            kind,
            initial,
            (),
            (false_context,),
            0,
            0,
            1,
            bool(evidence),
        )
    if kind is RolloutStressKind.MISSED_EVIDENCE_RECOVERY:
        frontier = (
            case.trace.rounds[-1].frontier_nodes
            if case.trace.rounds
            else initial
        )
        return RolloutStressExample(
            case,
            kind,
            frontier,
            (),
            (),
            max(0, len(case.trace.rounds) - 1),
            max(0, len(case.trace.rounds) - 1),
            0,
            bool(evidence),
        )
    if kind is RolloutStressKind.PREMATURE_STOP:
        return RolloutStressExample(
            case,
            kind,
            initial,
            (),
            (),
            0,
            0,
            0,
            bool(evidence),
        )
    if kind is RolloutStressKind.BUDGET_BOUNDARY:
        return RolloutStressExample(
            case,
            kind,
            initial,
            (),
            (),
            0,
            max(0, case.search_budget - 1),
            max(0, case.context_budget - 1),
            bool(evidence) and case.search_budget > 1,
        )
    return RolloutStressExample(
        case,
        kind,
        (*initial, *initial),
        (),
        (),
        0,
        0,
        0,
        bool(evidence),
    )


def generate_rollout_stress_examples(
    spec: SplitSpec,
) -> tuple[RolloutStressExample, ...]:
    if spec.name != "development_rollout_stress":
        raise ValueError("rollout stress examples require the stress split spec")
    cases = generate_split_cases(spec)
    kinds = tuple(RolloutStressKind)
    return tuple(
        _descriptor_for(case, kinds[index % len(kinds)])
        for index, case in enumerate(cases)
    )


def build_rollout_stress_manifest(
    spec: SplitSpec,
) -> RolloutStressManifest:
    descriptors = tuple(
        example.descriptor
        for example in generate_rollout_stress_examples(spec)
    )
    payload = json.dumps(
        {
            "spec": asdict(spec),
            "descriptors": descriptors,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return RolloutStressManifest(
        spec=spec,
        descriptors=descriptors,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
