from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


SRE_ADVERSARY_FAMILIES = (
    "query_echo",
    "answer_shaped",
    "stale_ownership",
    "rolled_back_deployment",
    "superseded_fix",
    "contradictory_memory",
    "same_entity_wrong_time",
    "same_entity_wrong_relationship",
    "near_duplicate",
    "relationship_path_decoy",
    "mixed_adversarial",
)

_RUNTIME_KEYS = frozenset(
    {
        "format",
        "case_id",
        "request_time",
        "query_text",
        "incoming_text",
        "candidates",
        "graph_edges",
    }
)
_CANDIDATE_KEYS = frozenset(
    {
        "candidate_id",
        "text",
        "memory_type",
        "occurred_at",
        "status",
        "region",
    }
)
_EDGE_KEYS = frozenset(
    {
        "source_memory_id",
        "destination_memory_id",
        "edge_type",
        "effective_at",
    }
)
_FORBIDDEN_RUNTIME_FIELDS = frozenset(
    {
        "positive_candidate_ids",
        "relevant",
        "adversary",
        "hard_negative",
        "relationship_path",
        "world_id",
        "groups",
        "provenance",
        "write_policy",
    }
)


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class SRECandidate:
    candidate_id: str
    text: str
    memory_type: str
    occurred_at: str
    status: str
    region: str

    @classmethod
    def from_dict(cls, value: Any) -> "SRECandidate":
        if not isinstance(value, dict) or set(value) != _CANDIDATE_KEYS:
            raise ValueError("candidate fields do not match the runtime schema")
        candidate = cls(
            candidate_id=_text(value["candidate_id"], "candidate_id"),
            text=_text(value["text"], "candidate text"),
            memory_type=_text(value["memory_type"], "memory_type"),
            occurred_at=_text(value["occurred_at"], "occurred_at"),
            status=_text(value["status"], "status"),
            region=_text(value["region"], "region"),
        )
        _timestamp(candidate.occurred_at, "occurred_at")
        return candidate


@dataclass(frozen=True, slots=True)
class SRERelationship:
    source_memory_id: str
    destination_memory_id: str
    edge_type: str
    effective_at: str

    @classmethod
    def from_dict(cls, value: Any) -> "SRERelationship":
        if not isinstance(value, dict) or set(value) != _EDGE_KEYS:
            raise ValueError("relationship fields do not match the runtime schema")
        edge = cls(
            source_memory_id=_text(value["source_memory_id"], "edge source"),
            destination_memory_id=_text(
                value["destination_memory_id"], "edge destination"
            ),
            edge_type=_text(value["edge_type"], "edge_type"),
            effective_at=_text(value["effective_at"], "effective_at"),
        )
        _timestamp(edge.effective_at, "effective_at")
        return edge


@dataclass(frozen=True, slots=True)
class SRECandidateLabel:
    candidate_id: str
    relevant: bool
    hard_negative: bool
    adversary: str | None


@dataclass(frozen=True, slots=True)
class SRERetrievalCase:
    case_id: str
    request_time: str
    query_text: str
    incoming_text: str | None
    candidates: tuple[SRECandidate, ...]
    relationships: tuple[SRERelationship, ...]
    labels: tuple[SRECandidateLabel, ...]
    scenario_family: str
    entity_lineage: str

    @property
    def pool_size(self) -> int:
        return len(self.candidates)

    @property
    def positive_candidate_ids(self) -> frozenset[str]:
        return frozenset(label.candidate_id for label in self.labels if label.relevant)

    @property
    def positive_count(self) -> int:
        return sum(label.relevant for label in self.labels)

    @property
    def relevance(self) -> tuple[bool, ...]:
        return tuple(label.relevant for label in self.labels)

    @classmethod
    def from_rows(
        cls,
        runtime: Any,
        supervision: Any,
        *,
        expected_pool_size: int | None = 64,
    ) -> "SRERetrievalCase":
        if not isinstance(runtime, dict) or set(runtime) != _RUNTIME_KEYS:
            raise ValueError("runtime input fields do not match the frozen schema")
        if any(field in runtime for field in _FORBIDDEN_RUNTIME_FIELDS):
            raise ValueError("runtime input contains a forbidden supervisor field")
        if not isinstance(supervision, dict):
            raise ValueError("supervision must be a JSON object")
        case_id = _text(runtime["case_id"], "case_id")
        if supervision.get("case_id") != case_id:
            raise ValueError("runtime and supervision case IDs disagree")
        request_time = _text(runtime["request_time"], "request_time")
        _timestamp(request_time, "request_time")
        incoming = runtime["incoming_text"]
        if incoming is not None:
            incoming = _text(incoming, "incoming_text")
        candidate_values = runtime["candidates"]
        if not isinstance(candidate_values, list):
            raise ValueError("candidates must be a list")
        if expected_pool_size is not None and len(candidate_values) != expected_pool_size:
            raise ValueError(
                f"candidate pool must contain exactly {expected_pool_size} records"
            )
        candidates = tuple(SRECandidate.from_dict(value) for value in candidate_values)
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within a case")
        candidate_set = set(candidate_ids)
        edge_values = runtime["graph_edges"]
        if not isinstance(edge_values, list):
            raise ValueError("graph_edges must be a list")
        relationships = tuple(
            SRERelationship.from_dict(value) for value in edge_values
        )
        if any(
            edge.source_memory_id not in candidate_set
            or edge.destination_memory_id not in candidate_set
            for edge in relationships
        ):
            raise ValueError("a relationship references a candidate outside the pool")

        label_values = supervision.get("candidate_labels")
        positive_values = supervision.get("positive_candidate_ids")
        groups = supervision.get("groups")
        if not isinstance(label_values, list) or not isinstance(positive_values, list):
            raise ValueError("candidate supervision is missing")
        if not isinstance(groups, dict):
            raise ValueError("supervision groups are missing")
        positive_ids = set(positive_values)
        if not positive_ids <= candidate_set:
            raise ValueError("positive IDs reference candidates outside the pool")
        labels_by_id: dict[str, SRECandidateLabel] = {}
        for value in label_values:
            if not isinstance(value, dict):
                raise ValueError("candidate label must be an object")
            candidate_id = _text(value.get("candidate_id"), "label candidate_id")
            relevant = value.get("relevant")
            hard_negative = value.get("hard_negative")
            adversary = value.get("adversary")
            if not isinstance(relevant, bool) or not isinstance(hard_negative, bool):
                raise ValueError("candidate label booleans are invalid")
            valid_adversary = (
                adversary == "positive"
                if relevant
                else adversary in SRE_ADVERSARY_FAMILIES
            )
            if not valid_adversary:
                raise ValueError(
                    f"candidate {candidate_id!r} has invalid adversary family "
                    f"{adversary!r} for relevant={relevant}"
                )
            if candidate_id in labels_by_id:
                raise ValueError("candidate supervision contains duplicates")
            labels_by_id[candidate_id] = SRECandidateLabel(
                candidate_id=candidate_id,
                relevant=relevant,
                hard_negative=hard_negative,
                adversary=None if relevant else adversary,
            )
        if set(labels_by_id) != candidate_set:
            raise ValueError("candidate supervision is not pool-aligned")
        labels = tuple(labels_by_id[candidate_id] for candidate_id in candidate_ids)
        if {label.candidate_id for label in labels if label.relevant} != positive_ids:
            raise ValueError("positive IDs and relevance labels disagree")
        return cls(
            case_id=case_id,
            request_time=request_time,
            query_text=_text(runtime["query_text"], "query_text"),
            incoming_text=incoming,
            candidates=candidates,
            relationships=relationships,
            labels=labels,
            scenario_family=_text(groups.get("scenario_family"), "scenario_family"),
            entity_lineage=_text(groups.get("entity_lineage"), "entity_lineage"),
        )


def _reject_sealed(path: Path) -> None:
    name = path.name.lower()
    if name.startswith("test.") or any(
        part.lower() in {"test", "test_sealed", "sealed_test"}
        for part in path.parts
    ):
        raise PermissionError(f"sealed SRE data is forbidden: {path}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _reject_sealed(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def load_sre_cases(
    inputs_path: str | Path,
    labels_path: str | Path,
    *,
    expected_pool_size: int | None = 64,
) -> tuple[SRERetrievalCase, ...]:
    inputs = _read_jsonl(Path(inputs_path))
    labels = _read_jsonl(Path(labels_path))
    labels_by_case = {row.get("case_id"): row for row in labels}
    if len(labels_by_case) != len(labels) or len(inputs) != len(labels):
        raise ValueError("input and label case cardinalities disagree")
    cases = tuple(
        SRERetrievalCase.from_rows(
            row,
            labels_by_case.get(row.get("case_id")),
            expected_pool_size=expected_pool_size,
        )
        for row in inputs
    )
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("duplicate SRE case IDs")
    return cases


def _stable_order(case: SRERetrievalCase, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}|{case.case_id}".encode()).digest()


def split_sre_development(
    cases: Sequence[SRERetrievalCase],
    *,
    selection_count: int,
    seed: int = 80_008,
) -> tuple[tuple[SRERetrievalCase, ...], tuple[SRERetrievalCase, ...]]:
    """Deterministically stratify by scenario and evidence cardinality."""

    if not 0 < selection_count < len(cases):
        raise ValueError("selection_count must split a non-empty development set")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("development cases must have unique IDs")
    groups: dict[tuple[str, int], list[SRERetrievalCase]] = defaultdict(list)
    for case in cases:
        groups[(case.scenario_family, case.positive_count)].append(case)
    total = len(cases)
    allocations: dict[tuple[str, int], int] = {}
    remainders: list[tuple[float, tuple[str, int]]] = []
    for key, members in groups.items():
        exact = len(members) * selection_count / total
        allocations[key] = int(exact)
        remainders.append((exact - int(exact), key))
    remaining = selection_count - sum(allocations.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        allocations[key] += 1
    selection: list[SRERetrievalCase] = []
    evaluation: list[SRERetrievalCase] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda case: _stable_order(case, seed))
        boundary = allocations[key]
        selection.extend(ordered[:boundary])
        evaluation.extend(ordered[boundary:])
    selection.sort(key=lambda case: _stable_order(case, seed + 1))
    evaluation.sort(key=lambda case: _stable_order(case, seed + 2))
    if len(selection) != selection_count:
        raise AssertionError("stratified selection cardinality drifted")
    return tuple(selection), tuple(evaluation)
