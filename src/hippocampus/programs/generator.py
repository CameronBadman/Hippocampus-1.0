from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from typing import Callable, Literal

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


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    min_nodes: int = 8
    max_nodes: int = 32
    min_path_length: int = 1
    max_path_length: int = 4
    min_summary_rows: int = 1
    max_summary_rows: int = 5
    min_context_rows: int = 0
    max_context_rows: int = 6
    max_distractor_edges: int = 8
    generator_version: str = "spider-programs-v0.1"
    lookup_binding_mode: Literal[
        "legacy_value_shared",
        "matched_conjunction",
    ] = "legacy_value_shared"

    def __post_init__(self) -> None:
        if self.min_nodes < 8:
            raise ValueError("Spider v0 generators require at least 8 nodes")
        if self.max_nodes < self.min_nodes:
            raise ValueError("max_nodes must be at least min_nodes")
        if self.min_path_length < 1:
            raise ValueError("min_path_length must be positive")
        if self.max_path_length < self.min_path_length:
            raise ValueError("max_path_length must be at least min_path_length")
        if self.min_summary_rows < 1:
            raise ValueError("summary manifolds may not be empty")
        if self.max_summary_rows < self.min_summary_rows:
            raise ValueError("invalid summary row range")
        if self.min_context_rows < 0:
            raise ValueError("context row counts must be non-negative")
        if self.max_context_rows < self.min_context_rows:
            raise ValueError("invalid context row range")
        if self.lookup_binding_mode not in {
            "legacy_value_shared",
            "matched_conjunction",
        }:
            raise ValueError("unsupported lookup binding mode")


@dataclass(slots=True)
class _MutableNode:
    latent_id: int
    name: str
    summary: list[ObservableAtom]
    context: list[ObservableAtom]


class GraphProgramGenerator:
    """Deterministic exact generator for the four Spider v0 task families."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()

    def generate(
        self,
        *,
        family: ProgramFamily | str,
        seed: int,
        answerable: bool | None = None,
        require_multiple_paths: bool = False,
        unknown_decision: TerminationDecision | str | None = None,
        context_budget_exhausted: bool = False,
    ) -> GraphProgramCase:
        resolved_family = ProgramFamily(family)
        resolved_unknown = (
            None
            if unknown_decision is None
            else TerminationDecision(unknown_decision)
        )
        if resolved_unknown in {
            TerminationDecision.CONTINUE,
            TerminationDecision.ANSWER,
        }:
            raise ValueError("unknown_decision must name an UNKNOWN outcome")
        resolved_answerable = (
            True
            if resolved_unknown is not None
            else seed % 2 == 0
            if answerable is None
            else answerable
        )
        rng_answerable = (
            False
            if resolved_family is ProgramFamily.LOOKUP
            and self.config.lookup_binding_mode == "matched_conjunction"
            else resolved_answerable
        )
        rng = random.Random(
            self._derived_seed(seed, resolved_family.value, rng_answerable)
        )
        dispatch: dict[ProgramFamily, Callable[..., GraphProgramCase]] = {
            ProgramFamily.LOOKUP: self._generate_lookup,
            ProgramFamily.REACHABILITY: self._generate_reachability,
            ProgramFamily.LATEST_VALID: self._generate_latest_valid,
            ProgramFamily.CORROBORATION: self._generate_corroboration,
        }
        case = dispatch[resolved_family](
            rng=rng,
            seed=seed,
            answerable=resolved_answerable,
            require_multiple_paths=require_multiple_paths,
        )
        if resolved_unknown is not None:
            case = self._force_unknown(
                case,
                decision=resolved_unknown,
                rng=rng,
                context_budget_exhausted=context_budget_exhausted,
            )
        return case

    def generate_suite(
        self,
        *,
        cases_per_family: int,
        seed: int,
    ) -> tuple[GraphProgramCase, ...]:
        if cases_per_family <= 0:
            raise ValueError("cases_per_family must be positive")
        cases: list[GraphProgramCase] = []
        for family_index, family in enumerate(ProgramFamily):
            for index in range(cases_per_family):
                case_seed = seed + family_index * 100_003 + index
                cases.append(
                    self.generate(
                        family=family,
                        seed=case_seed,
                        answerable=index % 2 == 0,
                        require_multiple_paths=(
                            family is ProgramFamily.REACHABILITY and index % 3 == 0
                        ),
                    )
                )
        return tuple(cases)

    @staticmethod
    def _derived_seed(seed: int, family: str, answerable: bool) -> int:
        digest = hashlib.sha256(
            f"{seed}|{family}|{int(answerable)}".encode()
        ).digest()
        return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)

    def _case_id(
        self,
        family: ProgramFamily,
        seed: int,
        answerable: bool,
    ) -> str:
        digest = hashlib.sha256(
            (
                f"{self.config.generator_version}|{family.value}|{seed}|"
                f"{int(answerable)}"
            ).encode()
        ).hexdigest()
        return f"{family.value}-{digest[:20]}"

    @staticmethod
    def _surface(rng: random.Random, prefix: str = "s") -> str:
        return f"{prefix}_{rng.getrandbits(64):016x}"

    @staticmethod
    def _latent_id(rng: random.Random, used: set[int]) -> int:
        while True:
            value = rng.randrange(1, 2**31)
            if value not in used:
                used.add(value)
                return value

    def _base_nodes(
        self,
        rng: random.Random,
        node_count: int,
    ) -> list[_MutableNode]:
        used: set[int] = set()
        nodes: list[_MutableNode] = []
        summary_span = (
            self.config.max_summary_rows - self.config.min_summary_rows + 1
        )
        context_span = (
            self.config.max_context_rows - self.config.min_context_rows + 1
        )
        for index in range(node_count):
            name = self._surface(rng, "n")
            summary = [ObservableAtom((name,))]
            target_summary_rows = self.config.min_summary_rows + index % summary_span
            while len(summary) < target_summary_rows:
                summary.append(ObservableAtom((self._surface(rng, "a"),)))

            context: list[ObservableAtom] = []
            target_context_rows = self.config.min_context_rows
            if context_span > 1 and index % 4 == 0:
                target_context_rows += index % context_span
            while len(context) < target_context_rows:
                context.append(ObservableAtom((self._surface(rng, "c"),)))
            nodes.append(
                _MutableNode(
                    latent_id=self._latent_id(rng, used),
                    name=name,
                    summary=summary,
                    context=context,
                )
            )
        return nodes

    @staticmethod
    def _freeze_nodes(nodes: list[_MutableNode]) -> tuple[ProgramNode, ...]:
        return tuple(
            ProgramNode(
                latent_id=node.latent_id,
                summary_atoms=tuple(node.summary),
                context_atoms=tuple(node.context),
            )
            for node in nodes
        )

    @staticmethod
    def _shuffle_edges(
        rng: random.Random,
        edges: list[ProgramEdge],
    ) -> tuple[ProgramEdge, ...]:
        rng.shuffle(edges)
        return tuple(edges)

    def _edge(
        self,
        rng: random.Random,
        used: set[int],
        source: int,
        destination: int,
        atoms: tuple[ObservableAtom, ...],
        *,
        valid: bool = True,
        bidirectional: bool = False,
    ) -> ProgramEdge:
        return ProgramEdge(
            latent_id=self._latent_id(rng, used),
            source_node=source,
            destination_node=destination,
            atoms=atoms,
            bidirectional=bidirectional,
            valid=valid,
        )

    def _force_unknown(
        self,
        case: GraphProgramCase,
        *,
        decision: TerminationDecision,
        rng: random.Random,
        context_budget_exhausted: bool,
    ) -> GraphProgramCase:
        if decision is TerminationDecision.UNKNOWN_CONFLICT:
            raise ValueError(
                "conflict outcomes must be generated by a family-specific oracle"
            )
        unsupported = decision is TerminationDecision.UNKNOWN_UNSUPPORTED
        rounds: list[OracleRound] = []
        for round_index, round_ in enumerate(case.trace.rounds):
            candidates = tuple(
                replace(
                    candidate,
                    acceptable=(
                        False if unsupported else candidate.acceptable
                    ),
                    context_has_value=(
                        False
                        if unsupported
                        else candidate.context_has_value
                    ),
                    include_as_evidence=False,
                    support=0.0,
                    conflict=0.0,
                )
                for candidate in round_.candidates
            )
            termination = (
                TerminationTarget(decision)
                if round_index == len(case.trace.rounds) - 1
                else TerminationTarget(TerminationDecision.CONTINUE)
            )
            rounds.append(
                OracleRound(round_.frontier_nodes, candidates, termination)
            )
            if unsupported:
                rounds[-1] = OracleRound(
                    round_.frontier_nodes,
                    candidates,
                    TerminationTarget(decision),
                )
                break
        query = case.query_atoms
        if unsupported:
            query = (
                *query,
                ObservableAtom((self._surface(rng, "unsupported"),)),
            )
        digest = hashlib.sha256(
            (
                f"{case.case_id}|{decision.value}|"
                f"{int(context_budget_exhausted)}"
            ).encode()
        ).hexdigest()
        termination = TerminationTarget(decision)
        return replace(
            case,
            case_id=f"{case.family.value}-{decision.value}-{digest[:20]}",
            base_case_id=f"{case.family.value}-{decision.value}-{digest[:20]}",
            view_id="base",
            query_atoms=query,
            answer_nodes=(),
            evidence_nodes=(),
            trace=ParallelOracleTrace(tuple(rounds), ()),
            termination=termination,
            search_budget=(
                case.search_budget
                if context_budget_exhausted
                else 0
                if decision is TerminationDecision.UNKNOWN_INCOMPLETE
                else case.search_budget
            ),
            context_budget=(
                0 if context_budget_exhausted else case.context_budget
            ),
        )

    def _add_background_edges(
        self,
        rng: random.Random,
        edges: list[ProgramEdge],
        used_edge_ids: set[int],
        nodes: list[_MutableNode],
        reserved_sources: set[int],
    ) -> None:
        existing = {
            (edge.source_node, edge.destination_node, edge.bidirectional)
            for edge in edges
        }
        target_count = min(
            self.config.max_distractor_edges,
            max(2, len(nodes) // 3),
        )
        attempts = 0
        while target_count > 0 and attempts < len(nodes) * 12:
            attempts += 1
            source = rng.randrange(len(nodes))
            destination = rng.randrange(len(nodes))
            if (
                source == destination
                or source in reserved_sources
                or (source, destination, False) in existing
            ):
                continue
            token = self._surface(rng, "r")
            edges.append(
                self._edge(
                    rng,
                    used_edge_ids,
                    source,
                    destination,
                    (ObservableAtom((token,)),),
                )
            )
            existing.add((source, destination, False))
            target_count -= 1

    def _final_case(
        self,
        *,
        family: ProgramFamily,
        seed: int,
        answerable: bool,
        nodes: list[_MutableNode],
        edges: tuple[ProgramEdge, ...],
        query_atoms: tuple[ObservableAtom, ...],
        start_nodes: tuple[int, ...],
        answer_nodes: tuple[int, ...],
        evidence_nodes: tuple[int, ...],
        trace: ParallelOracleTrace,
        termination: TerminationTarget,
        context_budget: int,
    ) -> GraphProgramCase:
        case_id = self._case_id(family, seed, answerable)
        return GraphProgramCase(
            case_id=case_id,
            base_case_id=case_id,
            view_id="base",
            seed=seed,
            family=family,
            nodes=self._freeze_nodes(nodes),
            edges=edges,
            query_atoms=query_atoms,
            start_nodes=start_nodes,
            answer_nodes=answer_nodes,
            evidence_nodes=evidence_nodes,
            trace=trace,
            termination=termination,
            search_budget=max(len(edges) * 2, len(nodes)),
            context_budget=context_budget,
        )

    def _generate_lookup(
        self,
        *,
        rng: random.Random,
        seed: int,
        answerable: bool,
        require_multiple_paths: bool,
    ) -> GraphProgramCase:
        if self.config.lookup_binding_mode == "matched_conjunction":
            return self._generate_matched_lookup(
                rng=rng,
                seed=seed,
                answerable=answerable,
                require_multiple_paths=require_multiple_paths,
            )
        del require_multiple_paths
        node_count = rng.randint(self.config.min_nodes, self.config.max_nodes)
        nodes = self._base_nodes(rng, node_count)
        roles = list(range(node_count))
        rng.shuffle(roles)
        start, correct, distractor_a, distractor_b = roles[:4]

        desired_value = self._surface(rng, "v")
        desired_relation = self._surface(rng, "r")
        other_relation = self._surface(rng, "r")
        for node_id in (correct, distractor_a, distractor_b):
            nodes[node_id].summary.append(ObservableAtom((desired_value,)))

        used_edges: set[int] = set()
        edges: list[ProgramEdge] = []
        decisive = self._edge(
            rng,
            used_edges,
            start,
            correct,
            (
                ObservableAtom(
                    (desired_relation if answerable else other_relation,)
                ),
            ),
        )
        edges.append(decisive)
        edges.append(
            self._edge(
                rng,
                used_edges,
                start,
                distractor_a,
                (ObservableAtom((other_relation,)),),
            )
        )
        edges.append(
            self._edge(
                rng,
                used_edges,
                start,
                distractor_b,
                (
                    ObservableAtom((desired_relation,)),
                    ObservableAtom(
                        (self._surface(rng, "gate"),),
                        scalar=-1.0,
                    ),
                ),
                valid=False,
            )
        )
        self._add_background_edges(rng, edges, used_edges, nodes, {start})
        frozen_edges = self._shuffle_edges(rng, edges)
        decisive_edge = next(
            index
            for index, edge in enumerate(frozen_edges)
            if edge.latent_id == decisive.latent_id
        )

        candidates = tuple(
            CandidateTarget(
                edge_id=index,
                source_node=edge.source_node,
                destination_node=edge.destination_node,
                acceptable=answerable and index == decisive_edge,
                priority_tier=0 if answerable and index == decisive_edge else 1,
                remaining_cost=0.0 if answerable and index == decisive_edge else 1.0,
                include_as_evidence=answerable and index == decisive_edge,
                support=1.0 if answerable and index == decisive_edge else 0.0,
            )
            for index, edge in enumerate(frozen_edges)
            if edge.source_node == start
        )
        answer_nodes = (correct,) if answerable else ()
        termination = TerminationTarget(
            TerminationDecision.ANSWER
            if answerable
            else TerminationDecision.UNKNOWN_ABSENT,
            answer_nodes,
        )
        trace = ParallelOracleTrace(
            rounds=(OracleRound((start,), candidates, termination),),
            valid_paths=((start, correct),) if answerable else (),
        )
        return self._final_case(
            family=ProgramFamily.LOOKUP,
            seed=seed,
            answerable=answerable,
            nodes=nodes,
            edges=frozen_edges,
            query_atoms=(
                ObservableAtom((nodes[start].name,)),
                ObservableAtom((desired_relation,)),
                ObservableAtom((desired_value,)),
            ),
            start_nodes=(start,),
            answer_nodes=answer_nodes,
            evidence_nodes=answer_nodes,
            trace=trace,
            termination=termination,
            context_budget=0,
        )

    def _generate_matched_lookup(
        self,
        *,
        rng: random.Random,
        seed: int,
        answerable: bool,
        require_multiple_paths: bool,
    ) -> GraphProgramCase:
        """Generate a metadata-matched relation/value conjunction task.

        Answerable and absent cases generated with the same seed share their
        topology, symbol inventory, row counts, and scalar inventory. The
        absent case breaks the sole valid relation/value conjunction while
        retaining each observable token elsewhere in the candidate set.
        """

        del require_multiple_paths
        node_count = rng.randint(self.config.min_nodes, self.config.max_nodes)
        nodes = self._base_nodes(rng, node_count)
        roles = list(range(node_count))
        rng.shuffle(roles)
        start, correct, wrong_value, wrong_relation, invalid = roles[:5]

        desired_value = self._surface(rng, "v")
        other_value = self._surface(rng, "v")
        desired_relation = self._surface(rng, "r")
        other_relation = self._surface(rng, "r")
        foreground = (correct, wrong_value, wrong_relation, invalid)
        target_base_rows = max(len(nodes[node_id].summary) for node_id in foreground)
        for node_id in foreground:
            while len(nodes[node_id].summary) < target_base_rows:
                nodes[node_id].summary.append(
                    ObservableAtom((self._surface(rng, "a"),))
                )
        destination_values = (
            desired_value if answerable else other_value,
            other_value,
            desired_value,
            desired_value,
        )
        for node_id, value in zip(
            foreground,
            destination_values,
            strict=True,
        ):
            nodes[node_id].summary.append(ObservableAtom((value,)))

        relations = (
            desired_relation if answerable else other_relation,
            desired_relation,
            other_relation,
            desired_relation,
        )
        validity = (True, True, True, False)
        gate_values = (1.0, 1.0, 1.0, -1.0)
        gate_tokens = tuple(self._surface(rng, "gate") for _ in foreground)
        used_edges: set[int] = set()
        edges: list[ProgramEdge] = []
        for node_id, relation, is_valid, gate_value, gate_token in zip(
            foreground,
            relations,
            validity,
            gate_values,
            gate_tokens,
            strict=True,
        ):
            edges.append(
                self._edge(
                    rng,
                    used_edges,
                    start,
                    node_id,
                    (
                        ObservableAtom((relation,)),
                        ObservableAtom((gate_token,), scalar=gate_value),
                    ),
                    valid=is_valid,
                )
            )
        decisive_latent_id = edges[0].latent_id
        self._add_background_edges(rng, edges, used_edges, nodes, {start})
        frozen_edges = self._shuffle_edges(rng, edges)
        decisive_edge = next(
            index
            for index, edge in enumerate(frozen_edges)
            if edge.latent_id == decisive_latent_id
        )

        candidates = tuple(
            CandidateTarget(
                edge_id=index,
                source_node=edge.source_node,
                destination_node=edge.destination_node,
                acceptable=answerable and index == decisive_edge,
                priority_tier=0 if answerable and index == decisive_edge else 1,
                remaining_cost=(
                    0.0 if answerable and index == decisive_edge else 1.0
                ),
                include_as_evidence=answerable and index == decisive_edge,
                support=1.0 if answerable and index == decisive_edge else 0.0,
            )
            for index, edge in enumerate(frozen_edges)
            if edge.source_node == start
        )
        answer_nodes = (correct,) if answerable else ()
        termination = TerminationTarget(
            TerminationDecision.ANSWER
            if answerable
            else TerminationDecision.UNKNOWN_ABSENT,
            answer_nodes,
        )
        trace = ParallelOracleTrace(
            rounds=(OracleRound((start,), candidates, termination),),
            valid_paths=((start, correct),) if answerable else (),
        )
        return self._final_case(
            family=ProgramFamily.LOOKUP,
            seed=seed,
            answerable=answerable,
            nodes=nodes,
            edges=frozen_edges,
            query_atoms=(
                ObservableAtom((nodes[start].name,)),
                ObservableAtom((desired_relation,)),
                ObservableAtom((desired_value,)),
            ),
            start_nodes=(start,),
            answer_nodes=answer_nodes,
            evidence_nodes=answer_nodes,
            trace=trace,
            termination=termination,
            context_budget=0,
        )

    def _generate_reachability(
        self,
        *,
        rng: random.Random,
        seed: int,
        answerable: bool,
        require_multiple_paths: bool,
    ) -> GraphProgramCase:
        path_length = rng.randint(
            self.config.min_path_length,
            self.config.max_path_length,
        )
        if require_multiple_paths:
            path_length = max(2, path_length)
        required_nodes = 2 + 2 * max(0, path_length - 1)
        node_count = rng.randint(
            max(self.config.min_nodes, required_nodes),
            max(self.config.max_nodes, required_nodes),
        )
        nodes = self._base_nodes(rng, node_count)
        roles = list(range(node_count))
        rng.shuffle(roles)
        start, target = roles[:2]
        cursor = 2

        path_a = [start]
        for _ in range(path_length - 1):
            path_a.append(roles[cursor])
            cursor += 1
        path_a.append(target)

        paths = [path_a]
        if require_multiple_paths or (answerable and path_length >= 2):
            path_b = [start]
            for _ in range(path_length - 1):
                path_b.append(roles[cursor])
                cursor += 1
            path_b.append(target)
            paths.append(path_b)

        pass_token = self._surface(rng, "r")
        gate_token = self._surface(rng, "g")
        used_edges: set[int] = set()
        edges: list[ProgramEdge] = []
        path_edge_latents: list[list[int]] = []
        for path in paths:
            latent_path: list[int] = []
            for source, destination in zip(path[:-1], path[1:], strict=True):
                edge = self._edge(
                    rng,
                    used_edges,
                    source,
                    destination,
                    (
                        ObservableAtom((pass_token,)),
                        ObservableAtom(
                            (gate_token,),
                            scalar=1.0 if answerable else -1.0,
                        ),
                    ),
                    valid=answerable,
                )
                edges.append(edge)
                latent_path.append(edge.latent_id)
            path_edge_latents.append(latent_path)

        unused = roles[cursor:]
        for destination in unused[: min(3, len(unused))]:
            edges.append(
                self._edge(
                    rng,
                    used_edges,
                    start,
                    destination,
                    (
                        ObservableAtom((pass_token,)),
                        ObservableAtom((gate_token,), scalar=-1.0),
                    ),
                    valid=False,
                )
            )
        self._add_background_edges(rng, edges, used_edges, nodes, set())
        frozen_edges = self._shuffle_edges(rng, edges)
        latent_to_edge = {
            edge.latent_id: index for index, edge in enumerate(frozen_edges)
        }

        answer_nodes = (target,) if answerable else ()
        termination = TerminationTarget(
            TerminationDecision.ANSWER
            if answerable
            else TerminationDecision.UNKNOWN_ABSENT,
            answer_nodes,
        )
        rounds: list[OracleRound] = []
        if answerable:
            for depth in range(path_length):
                frontier = tuple(dict.fromkeys(path[depth] for path in paths))
                acceptable_latents = {
                    path_edge_latents[path_index][depth]
                    for path_index in range(len(paths))
                }
                candidates = tuple(
                    CandidateTarget(
                        edge_id=index,
                        source_node=edge.source_node,
                        destination_node=edge.destination_node,
                        acceptable=edge.latent_id in acceptable_latents,
                        priority_tier=(
                            0 if edge.latent_id in acceptable_latents else 1
                        ),
                        remaining_cost=float(path_length - depth - 1),
                        include_as_evidence=(
                            edge.latent_id in acceptable_latents
                            and edge.destination_node == target
                        ),
                        support=(
                            1.0
                            if edge.latent_id in acceptable_latents
                            and edge.destination_node == target
                            else 0.0
                        ),
                    )
                    for index, edge in enumerate(frozen_edges)
                    if edge.source_node in frontier
                )
                round_termination = (
                    termination
                    if depth == path_length - 1
                    else TerminationTarget(TerminationDecision.CONTINUE)
                )
                rounds.append(OracleRound(frontier, candidates, round_termination))
        else:
            candidates = tuple(
                CandidateTarget(
                    edge_id=index,
                    source_node=edge.source_node,
                    destination_node=edge.destination_node,
                    acceptable=False,
                    priority_tier=1,
                    remaining_cost=float(path_length),
                )
                for index, edge in enumerate(frozen_edges)
                if edge.source_node == start
            )
            rounds.append(OracleRound((start,), candidates, termination))

        valid_paths = (
            tuple(tuple(path) for path in paths)
            if answerable
            else ()
        )
        trace = ParallelOracleTrace(tuple(rounds), valid_paths)
        return self._final_case(
            family=ProgramFamily.REACHABILITY,
            seed=seed,
            answerable=answerable,
            nodes=nodes,
            edges=frozen_edges,
            query_atoms=(
                ObservableAtom((nodes[start].name,)),
                ObservableAtom((nodes[target].name,)),
                ObservableAtom((pass_token,)),
                ObservableAtom((gate_token,), scalar=1.0),
            ),
            start_nodes=(start,),
            answer_nodes=answer_nodes,
            evidence_nodes=answer_nodes,
            trace=trace,
            termination=termination,
            context_budget=0,
        )

    def _generate_latest_valid(
        self,
        *,
        rng: random.Random,
        seed: int,
        answerable: bool,
        require_multiple_paths: bool,
    ) -> GraphProgramCase:
        del require_multiple_paths
        node_count = rng.randint(self.config.min_nodes, self.config.max_nodes)
        nodes = self._base_nodes(rng, node_count)
        roles = list(range(node_count))
        rng.shuffle(roles)
        start = roles[0]
        records = roles[1:4]

        subject_token = self._surface(rng, "subject")
        time_token = self._surface(rng, "measure")
        valid_token = self._surface(rng, "measure")
        browse_token = self._surface(rng, "r")
        value_tokens = [self._surface(rng, "v") for _ in records]

        if answerable:
            times = [1.0, 2.0, 3.0]
            validities = [1.0, 1.0, 1.0]
            answer_index = 2
            decision = TerminationDecision.ANSWER
            decisive_indices = {answer_index}
        elif seed % 2 == 0:
            times = [3.0, 3.0, 1.0]
            validities = [1.0, 1.0, 1.0]
            answer_index = -1
            decision = TerminationDecision.UNKNOWN_CONFLICT
            decisive_indices = {0, 1}
        else:
            times = [1.0, 2.0, 3.0]
            validities = [-1.0, -1.0, -1.0]
            answer_index = -1
            decision = TerminationDecision.UNKNOWN_ABSENT
            decisive_indices = set()

        for index, node_id in enumerate(records):
            nodes[node_id].summary.extend(
                (
                    ObservableAtom((subject_token,)),
                    ObservableAtom((value_tokens[index],)),
                )
            )
            nodes[node_id].context.extend(
                (
                    ObservableAtom((time_token,), scalar=times[index]),
                    ObservableAtom(
                        (valid_token,),
                        scalar=validities[index],
                    ),
                )
            )

        used_edges: set[int] = set()
        edges = [
            self._edge(
                rng,
                used_edges,
                start,
                record,
                (ObservableAtom((browse_token,)),),
            )
            for record in records
        ]
        self._add_background_edges(rng, edges, used_edges, nodes, {start})
        frozen_edges = self._shuffle_edges(rng, edges)

        record_to_index = {node: index for index, node in enumerate(records)}
        candidates = tuple(
            CandidateTarget(
                edge_id=edge_id,
                source_node=edge.source_node,
                destination_node=edge.destination_node,
                acceptable=(
                    edge.destination_node in record_to_index
                    and (
                        answerable
                        and record_to_index[edge.destination_node] == answer_index
                        or not answerable
                        and record_to_index[edge.destination_node]
                        in decisive_indices
                    )
                ),
                priority_tier=(
                    0
                    if edge.destination_node in record_to_index
                    and record_to_index[edge.destination_node] in decisive_indices
                    else 1
                ),
                remaining_cost=0.0,
                context_has_value=edge.destination_node in record_to_index,
                include_as_evidence=(
                    edge.destination_node in record_to_index
                    and record_to_index[edge.destination_node] in decisive_indices
                ),
                support=(
                    1.0
                    if answerable
                    and edge.destination_node in record_to_index
                    and record_to_index[edge.destination_node] == answer_index
                    else 0.0
                ),
                conflict=(
                    1.0
                    if decision is TerminationDecision.UNKNOWN_CONFLICT
                    and edge.destination_node in record_to_index
                    and record_to_index[edge.destination_node] in decisive_indices
                    else 0.0
                ),
            )
            for edge_id, edge in enumerate(frozen_edges)
            if edge.source_node == start
        )
        answer_nodes = (
            (records[answer_index],)
            if answer_index >= 0
            else ()
        )
        evidence_nodes = tuple(
            records[index] for index in sorted(decisive_indices)
        )
        termination = TerminationTarget(decision, answer_nodes)
        trace = ParallelOracleTrace(
            rounds=(OracleRound((start,), candidates, termination),),
            valid_paths=(
                tuple((start, node_id) for node_id in evidence_nodes)
            ),
        )
        return self._final_case(
            family=ProgramFamily.LATEST_VALID,
            seed=seed,
            answerable=answerable,
            nodes=nodes,
            edges=frozen_edges,
            query_atoms=(
                ObservableAtom((nodes[start].name,)),
                ObservableAtom((subject_token,)),
                ObservableAtom((time_token,), scalar=1.0),
                ObservableAtom((valid_token,), scalar=1.0),
            ),
            start_nodes=(start,),
            answer_nodes=answer_nodes,
            evidence_nodes=evidence_nodes,
            trace=trace,
            termination=termination,
            context_budget=len(records),
        )

    def _generate_corroboration(
        self,
        *,
        rng: random.Random,
        seed: int,
        answerable: bool,
        require_multiple_paths: bool,
    ) -> GraphProgramCase:
        del require_multiple_paths
        node_count = rng.randint(self.config.min_nodes, self.config.max_nodes)
        nodes = self._base_nodes(rng, node_count)
        roles = list(range(node_count))
        rng.shuffle(roles)
        start = roles[0]
        evidence_candidates = roles[1:5]
        claim_token = self._surface(rng, "claim")
        browse_token = self._surface(rng, "r")
        source_token = self._surface(rng, "source")

        if answerable:
            signs = [1.0, 1.0, 0.0, 0.0]
            decision = TerminationDecision.ANSWER
        elif seed % 2 == 0:
            signs = [1.0, -1.0, 0.0, 0.0]
            decision = TerminationDecision.UNKNOWN_CONFLICT
        else:
            signs = [1.0, 0.0, 0.0, 0.0]
            decision = TerminationDecision.UNKNOWN_ABSENT

        for index, node_id in enumerate(evidence_candidates):
            nodes[node_id].summary.append(
                ObservableAtom((claim_token,), scalar=signs[index])
            )
            nodes[node_id].context.append(
                ObservableAtom(
                    (source_token, self._surface(rng, "source_value")),
                    scalar=1.0,
                )
            )

        used_edges: set[int] = set()
        edges = [
            self._edge(
                rng,
                used_edges,
                start,
                node_id,
                (ObservableAtom((browse_token,)),),
            )
            for node_id in evidence_candidates
        ]
        self._add_background_edges(rng, edges, used_edges, nodes, {start})
        frozen_edges = self._shuffle_edges(rng, edges)
        sign_by_node = {
            node_id: signs[index]
            for index, node_id in enumerate(evidence_candidates)
        }
        selected_evidence = tuple(
            node_id
            for node_id in evidence_candidates
            if sign_by_node[node_id] != 0.0
        )
        answer_nodes = (
            tuple(
                node_id
                for node_id in evidence_candidates
                if sign_by_node[node_id] > 0
            )
            if answerable
            else ()
        )
        candidates = tuple(
            CandidateTarget(
                edge_id=edge_id,
                source_node=edge.source_node,
                destination_node=edge.destination_node,
                acceptable=(
                    edge.destination_node in sign_by_node
                    and sign_by_node[edge.destination_node] != 0.0
                ),
                priority_tier=(
                    0
                    if edge.destination_node in sign_by_node
                    and sign_by_node[edge.destination_node] != 0.0
                    else 1
                ),
                remaining_cost=0.0,
                include_as_evidence=(
                    edge.destination_node in sign_by_node
                    and sign_by_node[edge.destination_node] != 0.0
                ),
                support=max(sign_by_node.get(edge.destination_node, 0.0), 0.0),
                conflict=max(-sign_by_node.get(edge.destination_node, 0.0), 0.0),
            )
            for edge_id, edge in enumerate(frozen_edges)
            if edge.source_node == start
        )
        termination = TerminationTarget(decision, answer_nodes)
        trace = ParallelOracleTrace(
            rounds=(OracleRound((start,), candidates, termination),),
            valid_paths=tuple((start, node_id) for node_id in selected_evidence),
        )
        return self._final_case(
            family=ProgramFamily.CORROBORATION,
            seed=seed,
            answerable=answerable,
            nodes=nodes,
            edges=frozen_edges,
            query_atoms=(
                ObservableAtom((nodes[start].name,)),
                ObservableAtom((claim_token,)),
                ObservableAtom((source_token,)),
            ),
            start_nodes=(start,),
            answer_nodes=answer_nodes,
            evidence_nodes=selected_evidence,
            trace=trace,
            termination=termination,
            context_budget=0,
        )
