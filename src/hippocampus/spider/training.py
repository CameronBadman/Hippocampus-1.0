from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from ..programs import (
    GeneratorConfig,
    GraphProgramCase,
    GraphProgramGenerator,
    PackedProgramBatch,
    ProgramFamily,
    SyntheticManifoldRenderer,
    pack_rendered_cases,
)
from ..programs.schema import CandidateTarget, TerminationDecision
from .hypothesis import HypothesisBatch
from .controller import stable_candidate_selection
from .losses import (
    CandidateSupervision,
    LossTerm,
    SpiderLossConfig,
    SpiderLossReport,
    candidate_loss_report,
    termination_loss_term,
)
from .model import CandidateScorerBase
from .types import CandidateOutputs


TERMINATION_TO_INDEX = {
    TerminationDecision.CONTINUE: 0,
    TerminationDecision.ANSWER: 1,
    TerminationDecision.UNKNOWN_ABSENT: 2,
    TerminationDecision.UNKNOWN_CONFLICT: 3,
    TerminationDecision.UNKNOWN_INCOMPLETE: 4,
    TerminationDecision.UNKNOWN_UNSUPPORTED: 5,
}


@dataclass(frozen=True, slots=True)
class TrainingLoopConfig:
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.0
    seed: int = 0
    log_every: int = 25
    max_grad_norm: float = 5.0
    oracle_fraction_schedule: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not self.oracle_fraction_schedule:
            raise ValueError("oracle_fraction_schedule may not be empty")
        if any(
            fraction < 0.0 or fraction > 1.0
            for fraction in self.oracle_fraction_schedule
        ):
            raise ValueError("oracle fractions must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class OracleMetrics:
    candidate_count: int
    expand_correct: int
    context_correct: int
    evidence_correct: int
    support_correct: int
    conflict_correct: int
    priority_opportunities: int
    priority_top1_correct: int
    priority_reciprocal_rank_sum: float
    frontier_recall_at_4_sum: float
    predicted_expansions: int
    invalid_expansions: int
    context_true_positive: int
    context_false_positive: int
    context_false_negative: int
    evidence_true_positive: int
    evidence_false_positive: int
    evidence_false_negative: int
    termination_count: int
    termination_correct: int

    @property
    def expand_accuracy(self) -> float:
        return self.expand_correct / max(1, self.candidate_count)

    @property
    def context_accuracy(self) -> float:
        return self.context_correct / max(1, self.candidate_count)

    @property
    def evidence_accuracy(self) -> float:
        return self.evidence_correct / max(1, self.candidate_count)

    @property
    def support_accuracy(self) -> float:
        return self.support_correct / max(1, self.candidate_count)

    @property
    def conflict_accuracy(self) -> float:
        return self.conflict_correct / max(1, self.candidate_count)

    @property
    def priority_top1(self) -> float:
        return self.priority_top1_correct / max(1, self.priority_opportunities)

    @property
    def candidate_mrr(self) -> float:
        return self.priority_reciprocal_rank_sum / max(
            1,
            self.priority_opportunities,
        )

    @property
    def valid_frontier_recall_at_4(self) -> float:
        return self.frontier_recall_at_4_sum / max(
            1,
            self.priority_opportunities,
        )

    @property
    def invalid_expansion_rate(self) -> float:
        return self.invalid_expansions / max(1, self.predicted_expansions)

    @property
    def context_precision(self) -> float:
        return self.context_true_positive / max(
            1,
            self.context_true_positive + self.context_false_positive,
        )

    @property
    def context_recall(self) -> float:
        return self.context_true_positive / max(
            1,
            self.context_true_positive + self.context_false_negative,
        )

    @property
    def evidence_precision(self) -> float:
        return self.evidence_true_positive / max(
            1,
            self.evidence_true_positive + self.evidence_false_positive,
        )

    @property
    def evidence_recall(self) -> float:
        return self.evidence_true_positive / max(
            1,
            self.evidence_true_positive + self.evidence_false_negative,
        )

    @property
    def evidence_f1(self) -> float:
        precision = self.evidence_precision
        recall = self.evidence_recall
        return 2 * precision * recall / max(1e-12, precision + recall)

    @property
    def termination_accuracy(self) -> float:
        return self.termination_correct / max(1, self.termination_count)

    @property
    def joint_action_accuracy(self) -> float:
        return min(
            self.expand_accuracy,
            self.context_accuracy,
            self.evidence_accuracy,
            self.termination_accuracy,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "candidate_count": self.candidate_count,
            "candidate_expand_accuracy": self.expand_accuracy,
            "context_read_accuracy": self.context_accuracy,
            "evidence_inclusion_accuracy": self.evidence_accuracy,
            "support_accuracy": self.support_accuracy,
            "conflict_accuracy": self.conflict_accuracy,
            "candidate_top1": self.priority_top1,
            "candidate_mrr": self.candidate_mrr,
            "valid_frontier_recall_at_4": self.valid_frontier_recall_at_4,
            "invalid_expansion_rate": self.invalid_expansion_rate,
            "context_read_precision": self.context_precision,
            "context_read_recall": self.context_recall,
            "evidence_precision": self.evidence_precision,
            "evidence_recall": self.evidence_recall,
            "evidence_f1": self.evidence_f1,
            "termination_accuracy": self.termination_accuracy,
            "joint_action_accuracy": self.joint_action_accuracy,
        }

    def __add__(self, other: "OracleMetrics") -> "OracleMetrics":
        return OracleMetrics(
            *(
                getattr(self, field) + getattr(other, field)
                for field in self.__dataclass_fields__
            )
        )

    @classmethod
    def empty(cls) -> "OracleMetrics":
        return cls(*(0 for _ in cls.__dataclass_fields__))


@dataclass(frozen=True, slots=True)
class OracleRollout:
    loss: torch.Tensor
    report: SpiderLossReport
    metrics: OracleMetrics
    rounds: int


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    step: int
    loss: float
    gradient_norm: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    records: tuple[TrainingRecord, ...]
    initial_metrics: OracleMetrics
    final_metrics: OracleMetrics
    runtime_seconds: float


def _target_map(
    candidates: tuple[CandidateTarget, ...],
) -> dict[tuple[int, int, int], CandidateTarget]:
    return {
        (
            candidate.edge_id,
            candidate.source_node,
            candidate.destination_node,
        ): candidate
        for candidate in candidates
    }


def candidate_supervision(
    batch: PackedProgramBatch,
    round_index: int,
    expansion,
) -> CandidateSupervision:
    if batch.graph_count != 1:
        raise ValueError("oracle rollout currently consumes singleton packed cases")
    case = batch.cases[0]
    node_offset = int(batch.graph.topology.graph_node_ptr[0].item())
    edge_offset = int(batch.graph.topology.graph_edge_ptr[0].item())
    targets = _target_map(case.trace.rounds[round_index].candidates)
    resolved: list[CandidateTarget] = []
    for edge_id, source, destination in zip(
        expansion.edge_ids.tolist(),
        expansion.source_node_ids.tolist(),
        expansion.destination_node_ids.tolist(),
        strict=True,
    ):
        key = (
            edge_id - edge_offset,
            source - node_offset,
            destination - node_offset,
        )
        try:
            resolved.append(targets[key])
        except KeyError as exc:
            raise ValueError(
                f"expanded candidate {key} is absent from oracle round {round_index}"
            ) from exc
    device = batch.device
    return CandidateSupervision(
        acceptable=torch.tensor(
            [target.acceptable for target in resolved],
            dtype=torch.bool,
            device=device,
        ),
        context_has_value=torch.tensor(
            [target.context_has_value for target in resolved],
            dtype=torch.bool,
            device=device,
        ),
        include_as_evidence=torch.tensor(
            [target.include_as_evidence for target in resolved],
            dtype=torch.bool,
            device=device,
        ),
        remaining_cost=torch.tensor(
            [target.remaining_cost for target in resolved],
            dtype=torch.float32,
            device=device,
        ),
        support=torch.tensor(
            [target.support for target in resolved],
            dtype=torch.float32,
            device=device,
        ),
        conflict=torch.tensor(
            [target.conflict for target in resolved],
            dtype=torch.float32,
            device=device,
        ),
    )


def _encountered_supervision(
    batch: PackedProgramBatch,
    expansion,
) -> CandidateSupervision:
    case = batch.cases[0]
    node_offset = int(batch.graph.topology.graph_node_ptr[0].item())
    edge_offset = int(batch.graph.topology.graph_edge_ptr[0].item())
    targets = {
        (
            candidate.edge_id,
            candidate.source_node,
            candidate.destination_node,
        ): candidate
        for round_ in case.trace.rounds
        for candidate in round_.candidates
    }
    resolved: list[CandidateTarget] = []
    for edge_id, source, destination in zip(
        expansion.edge_ids.tolist(),
        expansion.source_node_ids.tolist(),
        expansion.destination_node_ids.tolist(),
        strict=True,
    ):
        key = (
            edge_id - edge_offset,
            source - node_offset,
            destination - node_offset,
        )
        resolved.append(
            targets.get(
                key,
                CandidateTarget(
                    edge_id=key[0],
                    source_node=key[1],
                    destination_node=key[2],
                    acceptable=False,
                    priority_tier=1,
                    remaining_cost=0.0,
                ),
            )
        )
    device = batch.device
    return CandidateSupervision(
        acceptable=torch.tensor(
            [target.acceptable for target in resolved],
            dtype=torch.bool,
            device=device,
        ),
        context_has_value=torch.tensor(
            [target.context_has_value for target in resolved],
            dtype=torch.bool,
            device=device,
        ),
        include_as_evidence=torch.tensor(
            [target.include_as_evidence for target in resolved],
            dtype=torch.bool,
            device=device,
        ),
        remaining_cost=torch.tensor(
            [target.remaining_cost for target in resolved],
            dtype=torch.float32,
            device=device,
        ),
        support=torch.tensor(
            [target.support for target in resolved],
            dtype=torch.float32,
            device=device,
        ),
        conflict=torch.tensor(
            [target.conflict for target in resolved],
            dtype=torch.float32,
            device=device,
        ),
    )


def _next_oracle_hypotheses(
    model: CandidateScorerBase,
    hypotheses: HypothesisBatch,
    expansion,
    outputs: CandidateOutputs,
    acceptable: torch.Tensor,
) -> HypothesisBatch:
    selected = torch.nonzero(acceptable, as_tuple=False).flatten()
    if selected.numel() == 0:
        return model.empty_hypotheses(hypotheses.device)
    parents = expansion.frontier_positions[selected].to(torch.int64)
    return HypothesisBatch(
        node_ids=expansion.destination_node_ids[selected],
        graph_ids=hypotheses.graph_ids[parents],
        path_state=outputs.next_path_state[selected],
        scores=hypotheses.scores[parents] + outputs.priority_logits[selected],
        depths=hypotheses.depths[parents] + 1,
        parent_trace_ids=torch.full(
            (selected.numel(),),
            -1,
            dtype=torch.int64,
            device=hypotheses.device,
        ),
        incoming_arc_ids=expansion.arc_ids[selected],
        incoming_edge_ids=expansion.edge_ids[selected],
        context_read=torch.zeros(
            selected.numel(),
            dtype=torch.bool,
            device=hypotheses.device,
        ),
    ).validate()


def _round_metrics(
    outputs: CandidateOutputs,
    context_logits: torch.Tensor,
    supervision: CandidateSupervision,
    frontier_positions: torch.Tensor,
    termination_logits: torch.Tensor,
    termination_target: int,
    *,
    frontier_count: int,
) -> OracleMetrics:
    count = supervision.count
    expand_correct = int(
        (
            (outputs.expand_logits >= 0) == supervision.acceptable
        ).sum().item()
    )
    context_correct = int(
        (
            (context_logits >= 0) == supervision.context_has_value
        ).sum().item()
    )
    evidence_correct = int(
        (
            (outputs.evidence_logits >= 0)
            == supervision.include_as_evidence
        ).sum().item()
    )
    support_correct = int(
        (
            (outputs.support_logits >= 0)
            == supervision.support.bool()
        ).sum().item()
    )
    conflict_correct = int(
        (
            (outputs.conflict_logits >= 0)
            == supervision.conflict.bool()
        ).sum().item()
    )
    priority_count = 0
    priority_correct = 0
    reciprocal_rank_sum = 0.0
    recall_at_4_sum = 0.0
    for frontier in range(frontier_count):
        positions = torch.nonzero(
            frontier_positions == frontier,
            as_tuple=False,
        ).flatten()
        if positions.numel() == 0:
            continue
        positives = positions[supervision.acceptable[positions]]
        if positives.numel() == 0:
            continue
        priority_count += 1
        order = positions[
            torch.argsort(
                outputs.priority_logits[positions],
                descending=True,
                stable=True,
            )
        ]
        winner = order[0]
        priority_correct += int(bool((positives == winner).any().item()))
        positive_ranks = torch.nonzero(
            supervision.acceptable[order],
            as_tuple=False,
        ).flatten()
        reciprocal_rank_sum += 1.0 / (int(positive_ranks[0].item()) + 1)
        recall_at_4_sum += float(
            supervision.acceptable[order[:4]].sum().item()
        ) / positives.numel()
    expand_predictions = outputs.expand_logits >= 0
    context_predictions = context_logits >= 0
    evidence_predictions = outputs.evidence_logits >= 0
    termination_prediction = int(termination_logits.argmax(dim=-1)[0].item())
    return OracleMetrics(
        candidate_count=count,
        expand_correct=expand_correct,
        context_correct=context_correct,
        evidence_correct=evidence_correct,
        support_correct=support_correct,
        conflict_correct=conflict_correct,
        priority_opportunities=priority_count,
        priority_top1_correct=priority_correct,
        priority_reciprocal_rank_sum=reciprocal_rank_sum,
        frontier_recall_at_4_sum=recall_at_4_sum,
        predicted_expansions=int(expand_predictions.sum().item()),
        invalid_expansions=int(
            (expand_predictions & ~supervision.acceptable).sum().item()
        ),
        context_true_positive=int(
            (
                context_predictions & supervision.context_has_value
            ).sum().item()
        ),
        context_false_positive=int(
            (
                context_predictions & ~supervision.context_has_value
            ).sum().item()
        ),
        context_false_negative=int(
            (
                ~context_predictions & supervision.context_has_value
            ).sum().item()
        ),
        evidence_true_positive=int(
            (
                evidence_predictions & supervision.include_as_evidence
            ).sum().item()
        ),
        evidence_false_positive=int(
            (
                evidence_predictions & ~supervision.include_as_evidence
            ).sum().item()
        ),
        evidence_false_negative=int(
            (
                ~evidence_predictions & supervision.include_as_evidence
            ).sum().item()
        ),
        termination_count=1,
        termination_correct=int(termination_prediction == termination_target),
    )


def _aggregate_reports(
    reports: Sequence[SpiderLossReport],
    *,
    reference: torch.Tensor,
) -> SpiderLossReport:
    names = tuple(reports[0].terms) if reports else ()
    aggregated: dict[str, LossTerm] = {}
    for name in names:
        terms = [report.terms[name] for report in reports]
        count = sum(term.target_count for term in terms)
        if count:
            raw = sum(
                term.raw * term.target_count for term in terms
            ) / count
        else:
            raw = reference.sum() * 0.0
        first = terms[0]
        if bool((first.raw.detach().abs() > 0).item()):
            weight = first.weighted.detach() / first.raw.detach()
            weighted = raw * weight
        else:
            weighted = sum(term.weighted for term in terms) / max(1, len(terms))
        aggregated[name] = LossTerm(raw, weighted, count)
    return SpiderLossReport(aggregated)


def oracle_rollout(
    model: CandidateScorerBase,
    batch: PackedProgramBatch,
    *,
    loss_config: SpiderLossConfig | None = None,
) -> OracleRollout:
    if batch.graph_count != 1:
        raise ValueError("oracle rollout currently requires one graph per batch")
    config = loss_config or SpiderLossConfig()
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    reports: list[SpiderLossReport] = []
    metrics = OracleMetrics.empty()
    case = batch.cases[0]
    for round_index, oracle_round in enumerate(case.trace.rounds):
        expansion = batch.graph.expand_frontier(
            hypotheses.node_ids,
            validate_ids=False,
        )
        supervision = candidate_supervision(batch, round_index, expansion)
        outputs = model.score_candidates(
            batch,
            hypotheses,
            expansion,
            evidence,
            round_index=round_index,
        )
        context_logits = outputs.context_logits
        context_candidates = torch.nonzero(
            supervision.context_has_value,
            as_tuple=False,
        ).flatten()
        if context_candidates.numel():
            outputs = model.refine_with_context(
                batch,
                expansion,
                outputs,
                context_candidates,
            )
        candidate_report = candidate_loss_report(
            outputs,
            supervision,
            expansion.frontier_positions,
            frontier_count=hypotheses.count,
            config=config,
            context_logits=context_logits,
        )
        next_hypotheses = _next_oracle_hypotheses(
            model,
            hypotheses,
            expansion,
            outputs,
            supervision.acceptable,
        )
        evidence_candidates = torch.nonzero(
            supervision.include_as_evidence,
            as_tuple=False,
        ).flatten()
        if evidence_candidates.numel():
            parents = expansion.frontier_positions[
                evidence_candidates
            ].to(torch.int64)
            evidence = model.update_evidence(
                evidence,
                outputs.next_path_state[evidence_candidates].mean(dim=1),
                hypotheses.graph_ids[parents],
            )
        termination_hypotheses = (
            next_hypotheses
            if next_hypotheses.count
            else hypotheses
        )
        termination_control = outputs.priority_logits.new_zeros((1, 6))
        termination_control[:, 0] = (round_index + 1) / max(
            1,
            len(case.trace.rounds),
        )
        termination_control[:, 3] = float(next_hypotheses.count == 0)
        termination_control[:, 4] = float(case.search_budget == 0)
        termination_control[:, 5] = float(
            case.context_budget == 0
            and bool(supervision.context_has_value.any().item())
        )
        termination_logits = model.termination_logits(
            batch,
            termination_hypotheses,
            evidence,
            termination_control,
        )
        termination_target = TERMINATION_TO_INDEX[
            oracle_round.termination.decision
        ]
        termination = termination_loss_term(
            termination_logits,
            torch.tensor(
                [termination_target],
                dtype=torch.int64,
                device=batch.device,
            ),
            config=config,
        )
        reports.append(
            SpiderLossReport(
                terms={
                    **candidate_report.terms,
                    "termination": termination,
                }
            )
        )
        metrics = metrics + _round_metrics(
            outputs,
            context_logits,
            supervision,
            expansion.frontier_positions,
            termination_logits,
            termination_target,
            frontier_count=hypotheses.count,
        )
        hypotheses = next_hypotheses
        if round_index + 1 < len(case.trace.rounds) and hypotheses.count == 0:
            raise ValueError("oracle trace continues after an empty acceptable frontier")

    report = _aggregate_reports(
        reports,
        reference=model.path_seed,
    )
    return OracleRollout(
        loss=report.total,
        report=report,
        metrics=metrics,
        rounds=len(case.trace.rounds),
    )


def mixed_rollout(
    model: CandidateScorerBase,
    batch: PackedProgramBatch,
    *,
    oracle_fraction: float,
    randomizer: random.Random,
    loss_config: SpiderLossConfig | None = None,
    max_rounds: int | None = None,
) -> OracleRollout:
    """Scheduled oracle/model execution with off-trace candidates as negatives."""

    if not 0.0 <= oracle_fraction <= 1.0:
        raise ValueError("oracle_fraction must be in [0, 1]")
    if batch.graph_count != 1:
        raise ValueError("mixed rollout currently requires singleton batches")
    config = loss_config or SpiderLossConfig()
    case = batch.cases[0]
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    reports: list[SpiderLossReport] = []
    metrics = OracleMetrics.empty()
    round_limit = max_rounds or max(2, len(case.trace.rounds) + 2)
    for rollout_round in range(round_limit):
        expansion = batch.graph.expand_frontier(
            hypotheses.node_ids,
            validate_ids=False,
        )
        supervision = _encountered_supervision(batch, expansion)
        outputs = model.score_candidates(
            batch,
            hypotheses,
            expansion,
            evidence,
            round_index=rollout_round,
        )
        context_logits = outputs.context_logits
        context_candidates = torch.nonzero(
            supervision.context_has_value,
            as_tuple=False,
        ).flatten()
        if context_candidates.numel():
            outputs = model.refine_with_context(
                batch,
                expansion,
                outputs,
                context_candidates,
            )
        candidate_report = candidate_loss_report(
            outputs,
            supervision,
            expansion.frontier_positions,
            frontier_count=hypotheses.count,
            config=config,
            context_logits=context_logits,
        )

        local_nodes = {
            node_id
            - int(batch.graph.topology.graph_node_ptr[0].item())
            for node_id in hypotheses.node_ids.tolist()
        }
        matching_round = next(
            (
                oracle_round
                for oracle_round in case.trace.rounds
                if set(oracle_round.frontier_nodes) == local_nodes
            ),
            None,
        )
        termination_decision = (
            matching_round.termination.decision
            if matching_round is not None
            else TerminationDecision.UNKNOWN_INCOMPLETE
        )
        termination_target = TERMINATION_TO_INDEX[termination_decision]
        termination_control = outputs.priority_logits.new_zeros((1, 6))
        termination_control[:, 0] = (rollout_round + 1) / round_limit
        termination_control[:, 4] = float(case.search_budget == 0)
        termination_control[:, 5] = float(
            case.context_budget == 0
            and bool(supervision.context_has_value.any().item())
        )
        termination_logits = model.termination_logits(
            batch,
            hypotheses,
            evidence,
            termination_control,
        )
        termination = termination_loss_term(
            termination_logits,
            torch.tensor(
                [termination_target],
                dtype=torch.int64,
                device=batch.device,
            ),
            config=config,
        )
        reports.append(
            SpiderLossReport(
                terms={
                    **candidate_report.terms,
                    "termination": termination,
                }
            )
        )
        metrics = metrics + _round_metrics(
            outputs,
            context_logits,
            supervision,
            expansion.frontier_positions,
            termination_logits,
            termination_target,
            frontier_count=hypotheses.count,
        )
        if (
            termination_decision is not TerminationDecision.CONTINUE
            or expansion.total_arcs == 0
        ):
            break

        use_oracle = (
            supervision.acceptable.any().item()
            and randomizer.random() < oracle_fraction
        )
        if use_oracle:
            selected = torch.nonzero(
                supervision.acceptable,
                as_tuple=False,
            ).flatten()
        else:
            selected = stable_candidate_selection(
                expansion,
                outputs.priority_logits
                + torch.nn.functional.logsigmoid(outputs.expand_logits),
                frontier_width=16,
                hypotheses_per_node=2,
            )
        selection_mask = torch.zeros(
            expansion.total_arcs,
            dtype=torch.bool,
            device=batch.device,
        )
        if selected.numel():
            selection_mask = selection_mask.index_fill(0, selected, True)
        evidence_candidates = selected[
            supervision.include_as_evidence[selected]
        ]
        if evidence_candidates.numel():
            parents = expansion.frontier_positions[
                evidence_candidates
            ].to(torch.int64)
            evidence = model.update_evidence(
                evidence,
                outputs.next_path_state[evidence_candidates].mean(dim=1),
                hypotheses.graph_ids[parents],
            )
        hypotheses = _next_oracle_hypotheses(
            model,
            hypotheses,
            expansion,
            outputs,
            selection_mask,
        )
        if hypotheses.count == 0:
            break

    report = _aggregate_reports(reports, reference=model.path_seed)
    return OracleRollout(
        loss=report.total,
        report=report,
        metrics=metrics,
        rounds=len(reports),
    )


def make_tiny_cases(
    *,
    case_count: int = 48,
    seed: int = 1101,
    generator_config: GeneratorConfig | None = None,
) -> tuple[GraphProgramCase, ...]:
    if case_count < 8:
        raise ValueError("tiny fixture needs at least eight cases")
    generator = GraphProgramGenerator(
        generator_config
        or GeneratorConfig(
            min_nodes=8,
            max_nodes=14,
            min_path_length=1,
            max_path_length=4,
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
        cases.append(
            generator.generate(
                family=family,
                seed=seed + index,
                answerable=answerable,
                require_multiple_paths=(
                    family is ProgramFamily.REACHABILITY and index % 3 == 0
                ),
                unknown_decision=unknown_decision,
                context_budget_exhausted=context_budget_exhausted,
            )
        )
    return tuple(cases)


def evaluate_oracle_batches(
    model: CandidateScorerBase,
    batches: Sequence[PackedProgramBatch],
    *,
    loss_config: SpiderLossConfig | None = None,
) -> tuple[float, OracleMetrics]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    metrics = OracleMetrics.empty()
    with torch.no_grad():
        for batch in batches:
            result = oracle_rollout(model, batch, loss_config=loss_config)
            total_loss += float(result.loss.item())
            metrics = metrics + result.metrics
    model.train(was_training)
    return total_loss / max(1, len(batches)), metrics


def train_oracle_batches(
    model: CandidateScorerBase,
    batches: Sequence[PackedProgramBatch],
    *,
    loop_config: TrainingLoopConfig,
    loss_config: SpiderLossConfig | None = None,
    checkpoint_path: Path | None = None,
) -> TrainingResult:
    if not batches:
        raise ValueError("training requires at least one packed batch")
    loss_settings = loss_config or SpiderLossConfig()
    randomizer = random.Random(loop_config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=loop_config.learning_rate,
        weight_decay=loop_config.weight_decay,
    )
    initial_loss, initial_metrics = evaluate_oracle_batches(
        model,
        batches,
        loss_config=loss_settings,
    )
    records = [
        TrainingRecord(
            step=0,
            loss=initial_loss,
            gradient_norm=0.0,
            elapsed_seconds=0.0,
        )
    ]
    started = time.perf_counter()
    model.train()
    for step in range(1, loop_config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        selected = [
            batches[randomizer.randrange(len(batches))]
            for _ in range(loop_config.batch_size)
        ]
        schedule_index = min(
            len(loop_config.oracle_fraction_schedule) - 1,
            (step - 1)
            * len(loop_config.oracle_fraction_schedule)
            // loop_config.steps,
        )
        oracle_fraction = loop_config.oracle_fraction_schedule[schedule_index]
        loss = torch.stack(
            [
                (
                    oracle_rollout(
                        model,
                        batch,
                        loss_config=loss_settings,
                    )
                    if oracle_fraction == 1.0
                    else mixed_rollout(
                        model,
                        batch,
                        oracle_fraction=oracle_fraction,
                        randomizer=randomizer,
                        loss_config=loss_settings,
                    )
                ).loss
                for batch in selected
            ]
        ).mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            loop_config.max_grad_norm,
        )
        optimizer.step()
        if step % loop_config.log_every == 0 or step == loop_config.steps:
            records.append(
                TrainingRecord(
                    step=step,
                    loss=float(loss.detach().item()),
                    gradient_norm=float(gradient_norm.detach().item()),
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
    runtime = time.perf_counter() - started
    final_loss, final_metrics = evaluate_oracle_batches(
        model,
        batches,
        loss_config=loss_settings,
    )
    records.append(
        TrainingRecord(
            step=loop_config.steps,
            loss=final_loss,
            gradient_norm=records[-1].gradient_norm,
            elapsed_seconds=runtime,
        )
    )
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": asdict(model.config),
                "loop_config": asdict(loop_config),
                "loss_config": asdict(loss_settings),
                "final_metrics": final_metrics.as_dict(),
            },
            checkpoint_path,
        )
    return TrainingResult(
        records=tuple(records),
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        runtime_seconds=runtime,
    )
