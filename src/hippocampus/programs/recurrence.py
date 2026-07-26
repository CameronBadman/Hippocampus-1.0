from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import random

from .oracle import verify_case
from .schema import (
    CandidateTarget,
    GraphProgramCase,
    ObservableAtom,
    OracleRound,
    ParallelOracleTrace,
    ProgramEdge,
    ProgramFamily,
    ProgramNode,
    TerminationDecision,
    TerminationTarget,
)


RECURRENCE_DATASET_VERSION = "spider-programs-v0.3-recurrence-dev"


@dataclass(frozen=True, slots=True)
class RecurrenceNecessitySpec:
    name: str
    case_count: int
    seed_start: int
    min_horizon: int = 4
    max_horizon: int = 8
    branch_count: int = 3
    dataset_version: str = RECURRENCE_DATASET_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("split name may not be empty")
        if self.case_count <= 0 or self.case_count % 2:
            raise ValueError("matched recurrence splits need a positive even size")
        if self.min_horizon < 4:
            raise ValueError("recurrence-necessity horizons begin at four")
        if self.max_horizon < self.min_horizon or self.max_horizon > 8:
            raise ValueError("recurrence-necessity horizons must be in [4, 8]")
        if self.branch_count < 3:
            raise ValueError("at least three matched branches are required")
        if self.dataset_version != RECURRENCE_DATASET_VERSION:
            raise ValueError("unexpected recurrence dataset version")


@dataclass(frozen=True, slots=True)
class RecurrenceNecessityManifest:
    spec: RecurrenceNecessitySpec
    case_ids: tuple[str, ...]
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "spec": asdict(self.spec),
            "case_ids": list(self.case_ids),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class RecurrenceLeakageReport:
    case_count: int
    branch_count: int
    chance_accuracy: float
    final_position_accuracy: float
    first_hop_profile_mismatch_count: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def default_recurrence_necessity_specs() -> tuple[
    RecurrenceNecessitySpec,
    ...,
]:
    return (
        RecurrenceNecessitySpec(
            name="train_recurrence_necessity",
            case_count=512,
            seed_start=810_000,
        ),
        RecurrenceNecessitySpec(
            name="validation_recurrence_necessity",
            case_count=128,
            seed_start=820_000,
        ),
    )


def _surface(rng: random.Random, prefix: str) -> str:
    return f"{prefix}_{rng.getrandbits(64):016x}"


def _latent_id(rng: random.Random, used: set[int]) -> int:
    while True:
        value = rng.randrange(1, 2**31)
        if value not in used:
            used.add(value)
            return value


def _one_fixed_point_assignment(
    tokens: tuple[str, ...],
    fixed_index: int,
) -> tuple[str, ...]:
    remaining = [
        index for index in range(len(tokens)) if index != fixed_index
    ]
    assignment = list(tokens)
    rotated = remaining[1:] + remaining[:1]
    for destination, source in zip(remaining, rotated, strict=True):
        assignment[destination] = tokens[source]
    assignment[fixed_index] = tokens[fixed_index]
    return tuple(assignment)


def _case_from_assignment(
    *,
    seed: int,
    horizon: int,
    branch_count: int,
    view_id: str,
    pair_id: str,
    nodes: tuple[ProgramNode, ...],
    edge_templates: tuple[
        tuple[int, int, int, tuple[ObservableAtom, ...]],
        ...,
    ],
    first_edge_latents: tuple[int, ...],
    final_edge_latents: tuple[int, ...],
    first_assignment: tuple[str, ...],
    correct_branch: int,
    edge_order: tuple[int, ...],
    query_atoms: tuple[ObservableAtom, ...],
) -> GraphProgramCase:
    assignment_by_latent = dict(
        zip(first_edge_latents, first_assignment, strict=True)
    )
    edges_unordered = tuple(
        ProgramEdge(
            latent_id=latent_id,
            source_node=source,
            destination_node=destination,
            atoms=(
                (ObservableAtom((assignment_by_latent[latent_id],)),)
                if latent_id in assignment_by_latent
                else atoms
            ),
        )
        for latent_id, source, destination, atoms in edge_templates
    )
    edges = tuple(edges_unordered[index] for index in edge_order)
    latent_to_edge = {
        edge.latent_id: edge_id for edge_id, edge in enumerate(edges)
    }
    path_nodes = tuple(
        tuple(1 + branch * (horizon - 1) + depth for depth in range(horizon - 1))
        for branch in range(branch_count)
    )
    sink = len(nodes) - 1
    path_edge_ids: list[tuple[int, ...]] = []
    for branch in range(branch_count):
        branch_latents = [first_edge_latents[branch]]
        branch_latents.extend(
            latent_id
            for latent_id, source, _, _ in edge_templates
            if (
                source in path_nodes[branch][:-1]
                and latent_id not in final_edge_latents
            )
        )
        branch_latents.sort(
            key=lambda latent: edges_unordered[
                next(
                    index
                    for index, template in enumerate(edge_templates)
                    if template[0] == latent
                )
            ].source_node
        )
        branch_latents.append(final_edge_latents[branch])
        path_edge_ids.append(
            tuple(latent_to_edge[latent] for latent in branch_latents)
        )

    evidence_edge_id = latent_to_edge[final_edge_latents[correct_branch]]
    termination = TerminationTarget(
        TerminationDecision.ANSWER,
        (sink,),
    )
    rounds: list[OracleRound] = []
    for depth in range(horizon):
        frontier = (
            (0,)
            if depth == 0
            else tuple(path_nodes[branch][depth - 1] for branch in range(branch_count))
        )
        candidates: list[CandidateTarget] = []
        for branch in range(branch_count):
            edge_id = path_edge_ids[branch][depth]
            edge = edges[edge_id]
            include = edge_id == evidence_edge_id and depth == horizon - 1
            candidates.append(
                CandidateTarget(
                    edge_id=edge_id,
                    source_node=edge.source_node,
                    destination_node=edge.destination_node,
                    acceptable=True,
                    priority_tier=0 if include else 1 if depth == horizon - 1 else 0,
                    remaining_cost=float(horizon - depth - 1),
                    include_as_evidence=include,
                    support=1.0 if include else 0.0,
                )
            )
        rounds.append(
            OracleRound(
                frontier_nodes=frontier,
                candidates=tuple(candidates),
                termination=(
                    termination
                    if depth == horizon - 1
                    else TerminationTarget(TerminationDecision.CONTINUE)
                ),
            )
        )

    correct_nodes = (
        0,
        *path_nodes[correct_branch],
        sink,
    )
    case_digest = hashlib.sha256(
        (
            f"{RECURRENCE_DATASET_VERSION}|{seed}|{horizon}|"
            f"{view_id}|{correct_branch}"
        ).encode()
    ).hexdigest()[:20]
    case = GraphProgramCase(
        case_id=f"recurrence-{case_digest}",
        base_case_id=pair_id,
        view_id=view_id,
        seed=seed,
        family=ProgramFamily.REACHABILITY,
        nodes=nodes,
        edges=edges,
        query_atoms=query_atoms,
        start_nodes=(0,),
        answer_nodes=(sink,),
        evidence_nodes=(sink,),
        trace=ParallelOracleTrace(
            rounds=tuple(rounds),
            valid_paths=(correct_nodes,),
        ),
        termination=termination,
        search_budget=max(len(edges) * 2, len(nodes)),
        context_budget=0,
        evidence_edge_ids=(evidence_edge_id,),
    )
    verify_case(case).raise_for_errors()
    return case


def generate_recurrence_necessity_pair(
    *,
    seed: int,
    horizon: int,
    branch_count: int = 3,
) -> tuple[GraphProgramCase, GraphProgramCase]:
    """Generate a matched pair solvable only by cross-hop token comparison."""

    if not 4 <= horizon <= 8:
        raise ValueError("recurrence diagnostic horizon must be in [4, 8]")
    if branch_count < 3:
        raise ValueError("at least three branches are required")
    rng = random.Random(
        int.from_bytes(
            hashlib.sha256(
                f"{RECURRENCE_DATASET_VERSION}|{seed}|{horizon}".encode()
            ).digest()[:8],
            "little",
        )
    )
    nonce = _surface(rng, "case")
    used_node_ids: set[int] = set()
    node_count = 2 + branch_count * (horizon - 1)
    nodes: list[ProgramNode] = [
        ProgramNode(
            latent_id=_latent_id(rng, used_node_ids),
            summary_atoms=(ObservableAtom((f"{nonce}_start",)),),
        )
    ]
    for _branch in range(branch_count):
        for depth in range(1, horizon):
            nodes.append(
                ProgramNode(
                    latent_id=_latent_id(rng, used_node_ids),
                    summary_atoms=(
                        ObservableAtom((f"{nonce}_level_{depth}",)),
                    ),
                )
            )
    nodes.append(
        ProgramNode(
            latent_id=_latent_id(rng, used_node_ids),
            summary_atoms=(ObservableAtom((f"{nonce}_sink",)),),
        )
    )
    frozen_nodes = tuple(nodes)
    sink = len(frozen_nodes) - 1
    comparison_tokens = tuple(
        _surface(rng, "bind") for _ in range(branch_count)
    )
    used_edge_ids: set[int] = set()
    templates: list[
        tuple[int, int, int, tuple[ObservableAtom, ...]]
    ] = []
    first_edge_latents: list[int] = []
    final_edge_latents: list[int] = []
    for branch in range(branch_count):
        branch_nodes = [
            1 + branch * (horizon - 1) + depth
            for depth in range(horizon - 1)
        ]
        first_latent = _latent_id(rng, used_edge_ids)
        first_edge_latents.append(first_latent)
        templates.append(
            (
                first_latent,
                0,
                branch_nodes[0],
                (ObservableAtom((comparison_tokens[branch],)),),
            )
        )
        for depth, (source, destination) in enumerate(
            zip(branch_nodes[:-1], branch_nodes[1:], strict=True),
            start=1,
        ):
            templates.append(
                (
                    _latent_id(rng, used_edge_ids),
                    source,
                    destination,
                    (ObservableAtom((f"{nonce}_step_{depth}",)),),
                )
            )
        final_latent = _latent_id(rng, used_edge_ids)
        final_edge_latents.append(final_latent)
        templates.append(
            (
                final_latent,
                branch_nodes[-1],
                sink,
                (ObservableAtom((comparison_tokens[branch],)),),
            )
        )
    edge_order_list = list(range(len(templates)))
    rng.shuffle(edge_order_list)
    edge_order = tuple(edge_order_list)
    correct_left = rng.randrange(branch_count)
    correct_right = rng.choice(
        [index for index in range(branch_count) if index != correct_left]
    )
    pair_id = "recurrence-pair-" + hashlib.sha256(
        f"{RECURRENCE_DATASET_VERSION}|{seed}|{horizon}|pair".encode()
    ).hexdigest()[:20]
    query_atoms = (
        ObservableAtom((f"{nonce}_query",)),
        ObservableAtom(scalar=1.0),
    )
    common = {
        "seed": seed,
        "horizon": horizon,
        "branch_count": branch_count,
        "pair_id": pair_id,
        "nodes": frozen_nodes,
        "edge_templates": tuple(templates),
        "first_edge_latents": tuple(first_edge_latents),
        "final_edge_latents": tuple(final_edge_latents),
        "edge_order": edge_order,
        "query_atoms": query_atoms,
    }
    return (
        _case_from_assignment(
            **common,
            view_id="assignment_a",
            first_assignment=_one_fixed_point_assignment(
                comparison_tokens,
                correct_left,
            ),
            correct_branch=correct_left,
        ),
        _case_from_assignment(
            **common,
            view_id="assignment_b",
            first_assignment=_one_fixed_point_assignment(
                comparison_tokens,
                correct_right,
            ),
            correct_branch=correct_right,
        ),
    )


def generate_recurrence_necessity_cases(
    spec: RecurrenceNecessitySpec,
    *,
    limit: int | None = None,
) -> tuple[GraphProgramCase, ...]:
    count = spec.case_count if limit is None else min(limit, spec.case_count)
    if count <= 0 or count % 2:
        raise ValueError("recurrence case limits must be positive and even")
    horizon_span = spec.max_horizon - spec.min_horizon + 1
    cases: list[GraphProgramCase] = []
    for pair_index in range(count // 2):
        cases.extend(
            generate_recurrence_necessity_pair(
                seed=spec.seed_start + pair_index,
                horizon=spec.min_horizon + pair_index % horizon_span,
                branch_count=spec.branch_count,
            )
        )
    return tuple(cases)


def build_recurrence_necessity_manifest(
    spec: RecurrenceNecessitySpec,
) -> RecurrenceNecessityManifest:
    cases = generate_recurrence_necessity_cases(spec)
    case_ids = tuple(case.case_id for case in cases)
    payload = json.dumps(
        {
            "spec": asdict(spec),
            "case_ids": case_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return RecurrenceNecessityManifest(
        spec=spec,
        case_ids=case_ids,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def recurrence_metadata_leakage_report(
    cases: tuple[GraphProgramCase, ...],
) -> RecurrenceLeakageReport:
    if not cases:
        raise ValueError("leakage diagnostics require cases")
    position_counts: Counter[int] = Counter()
    mismatch_count = 0
    branch_counts: set[int] = set()
    for case in cases:
        first = case.trace.rounds[0]
        final = case.trace.rounds[-1]
        branch_counts.add(len(first.candidates))
        positive_positions = [
            index
            for index, candidate in enumerate(final.candidates)
            if candidate.include_as_evidence
        ]
        if len(positive_positions) != 1:
            raise ValueError("recurrence cases need one exact evidence action")
        position_counts[positive_positions[0]] += 1
        first_profiles = {
            (
                case.nodes[candidate.destination_node].summary_atoms,
                sum(
                    edge.source_node == candidate.destination_node
                    for edge in case.edges
                ),
                len(case.edges[candidate.edge_id].atoms),
            )
            for candidate in first.candidates
        }
        mismatch_count += int(len(first_profiles) != 1)
    if len(branch_counts) != 1:
        raise ValueError("all leakage cases must use one branch count")
    branch_count = branch_counts.pop()
    return RecurrenceLeakageReport(
        case_count=len(cases),
        branch_count=branch_count,
        chance_accuracy=1.0 / branch_count,
        final_position_accuracy=max(position_counts.values()) / len(cases),
        first_hop_profile_mismatch_count=mismatch_count,
    )

