from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


class ProgramFamily(str, Enum):
    LOOKUP = "lookup"
    REACHABILITY = "reachability"
    LATEST_VALID = "latest_valid"
    CORROBORATION = "corroboration"


class TerminationDecision(str, Enum):
    CONTINUE = "continue"
    ANSWER = "answer"
    UNKNOWN_ABSENT = "unknown_absent"
    UNKNOWN_CONFLICT = "unknown_conflict"
    UNKNOWN_INCOMPLETE = "unknown_incomplete"
    UNKNOWN_UNSUPPORTED = "unknown_unsupported"


class CounterfactualKind(str, Enum):
    REMOVE_DECISIVE_EDGE = "remove_decisive_edge"
    REVERSE_DECISIVE_EDGE = "reverse_decisive_edge"
    ALTER_TEMPORAL_VALUE = "alter_temporal_value"
    ADD_NEWER_CONFLICT = "add_newer_conflict"
    INVALIDATE_SOURCE = "invalidate_source"
    REPLACE_ENDPOINT = "replace_endpoint"
    DISCONNECT_ONLY_PATH = "disconnect_only_path"


Scalar: TypeAlias = float | int


@dataclass(frozen=True, slots=True)
class ObservableAtom:
    """One unordered model-visible symbolic or scalar observation.

    Symbols are opaque surface codes. A scalar is an observable measurement,
    not a fixed feature slot. Supervisor labels deliberately do not fit this
    schema.
    """

    symbols: tuple[str, ...] = ()
    scalar: float | None = None

    def __post_init__(self) -> None:
        if not self.symbols and self.scalar is None:
            raise ValueError("an observable atom must contain a symbol or scalar")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("observable atom symbols must be unique")


@dataclass(frozen=True, slots=True)
class ProgramNode:
    latent_id: int
    summary_atoms: tuple[ObservableAtom, ...]
    context_atoms: tuple[ObservableAtom, ...] = ()

    def __post_init__(self) -> None:
        if not self.summary_atoms:
            raise ValueError("program node summaries may not be empty")


@dataclass(frozen=True, slots=True)
class ProgramEdge:
    latent_id: int
    source_node: int
    destination_node: int
    atoms: tuple[ObservableAtom, ...]
    bidirectional: bool = False
    valid: bool = True

    def __post_init__(self) -> None:
        if not self.atoms:
            raise ValueError("program edge observations may not be empty")


@dataclass(frozen=True, slots=True)
class CandidateTarget:
    edge_id: int
    source_node: int
    destination_node: int
    acceptable: bool
    priority_tier: int = 0
    remaining_cost: float = 0.0
    context_has_value: bool = False
    include_as_evidence: bool = False
    support: float = 0.0
    conflict: float = 0.0


@dataclass(frozen=True, slots=True)
class TerminationTarget:
    decision: TerminationDecision
    answer_nodes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.decision is TerminationDecision.ANSWER and not self.answer_nodes:
            raise ValueError("ANSWER termination requires at least one answer node")
        if self.decision is not TerminationDecision.ANSWER and self.answer_nodes:
            raise ValueError("unknown/continue termination may not carry answer nodes")


@dataclass(frozen=True, slots=True)
class OracleRound:
    frontier_nodes: tuple[int, ...]
    candidates: tuple[CandidateTarget, ...]
    termination: TerminationTarget


@dataclass(frozen=True, slots=True)
class ParallelOracleTrace:
    rounds: tuple[OracleRound, ...]
    valid_paths: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class Intervention:
    kind: CounterfactualKind
    changed_latent_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GraphProgramCase:
    case_id: str
    base_case_id: str
    view_id: str
    seed: int
    family: ProgramFamily
    nodes: tuple[ProgramNode, ...]
    edges: tuple[ProgramEdge, ...]
    query_atoms: tuple[ObservableAtom, ...]
    start_nodes: tuple[int, ...]
    answer_nodes: tuple[int, ...]
    evidence_nodes: tuple[int, ...]
    trace: ParallelOracleTrace
    termination: TerminationTarget
    search_budget: int
    context_budget: int
    intervention: Intervention | None = None

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("a graph-program case requires at least one node")
        if not self.query_atoms:
            raise ValueError("a graph-program case requires a query manifold")
        if not self.start_nodes:
            raise ValueError("a graph-program case requires a start node")
        if self.trace.rounds and self.trace.rounds[-1].termination != self.termination:
            raise ValueError("case termination must match the final oracle round")
        if self.termination.answer_nodes != self.answer_nodes:
            raise ValueError("case answer nodes must match termination target")

    @property
    def answerable(self) -> bool:
        return self.termination.decision is TerminationDecision.ANSWER

    @property
    def answer_latent_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.nodes[index].latent_id for index in self.answer_nodes))

    @property
    def evidence_latent_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.nodes[index].latent_id for index in self.evidence_nodes))

    @property
    def latent_edge_ids(self) -> tuple[int, ...]:
        return tuple(sorted(edge.latent_id for edge in self.edges))
