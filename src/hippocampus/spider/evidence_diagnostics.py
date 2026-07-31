from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch

from ..programs.batching import PackedProgramBatch
from .calibration import binary_average_precision
from .controller import ControllerResult, EvidenceLedgerEntry
from .state_oracle import EvidenceRequirement, StateOracle


@dataclass(frozen=True, slots=True)
class EvidenceRequirementObservation:
    requirement: EvidenceRequirement
    round_index: int
    outstanding_before: bool
    reachable: bool
    enumerated: bool
    scored: bool
    selected: bool
    recorded: bool
    frontier_selected: bool
    candidate_count: int
    best_logit: float | None
    best_rank: int | None
    positive_negative_margin: float | None


@dataclass(frozen=True, slots=True)
class EvidenceCandidateObservation:
    round_index: int
    arc_id: int
    edge_id: int
    source_node: int
    destination_node: int
    logit: float
    pre_context_logit: float
    required: bool
    selected: bool
    recorded: bool
    frontier_selected: bool


@dataclass(frozen=True, slots=True)
class EvidencePipelineCaseReport:
    case_id: str
    family: str
    horizon: int
    requirements: tuple[EvidenceRequirement, ...]
    requirement_observations: tuple[EvidenceRequirementObservation, ...]
    candidate_observations: tuple[EvidenceCandidateObservation, ...]
    exact_set_accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int
    predicted_cardinality: int
    required_cardinality: int
    average_precision: float
    worst_positive_rank: int | None
    minimum_positive_negative_margin: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _matches(
    requirement: EvidenceRequirement,
    *,
    edge_id: int,
    source_node: int,
    destination_node: int,
) -> bool:
    if destination_node != requirement.destination_node:
        return False
    if requirement.edge_id is None:
        return True
    return (
        edge_id == requirement.edge_id
        and source_node == requirement.source_node
    )


def _matching_indices(
    requirement: EvidenceRequirement,
    edge_ids: torch.Tensor,
    source_ids: torch.Tensor,
    destination_ids: torch.Tensor,
    *,
    edge_offset: int,
    node_offset: int,
) -> torch.Tensor:
    destination = (
        destination_ids.to(torch.int64) - node_offset
        == requirement.destination_node
    )
    if requirement.edge_id is None:
        return torch.nonzero(destination, as_tuple=False).flatten()
    exact = (
        destination
        & (
            edge_ids.to(torch.int64) - edge_offset
            == requirement.edge_id
        )
        & (
            source_ids.to(torch.int64) - node_offset
            == requirement.source_node
        )
    )
    return torch.nonzero(exact, as_tuple=False).flatten()


def _entry_matches(
    requirement: EvidenceRequirement,
    entry: EvidenceLedgerEntry,
    *,
    edge_offset: int,
    node_offset: int,
) -> bool:
    if entry.node_id - node_offset != requirement.destination_node:
        return False
    return (
        requirement.edge_id is None
        or entry.edge_id - edge_offset == requirement.edge_id
    )


def _predicted_keys(
    batch: PackedProgramBatch,
    entries: Sequence[EvidenceLedgerEntry],
    *,
    edge_specific: bool,
) -> set[tuple[int | None, int | None, int]]:
    case = batch.cases[0]
    edge_offset = int(batch.graph.topology.graph_edge_ptr[0].item())
    node_offset = int(batch.graph.topology.graph_node_ptr[0].item())
    keys: set[tuple[int | None, int | None, int]] = set()
    for entry in entries:
        destination = entry.node_id - node_offset
        if not edge_specific:
            keys.add((None, None, destination))
            continue
        local_edge = entry.edge_id - edge_offset
        source = (
            case.edges[local_edge].source_node
            if 0 <= local_edge < len(case.edges)
            else None
        )
        keys.add((local_edge, source, destination))
    return keys


def observe_evidence_pipeline(
    batch: PackedProgramBatch,
    result: ControllerResult,
    oracle: StateOracle,
) -> EvidencePipelineCaseReport:
    """Join model-policy round records with supervisor-only evidence truth."""

    if batch.graph_count != 1:
        raise ValueError("evidence diagnostics require a singleton graph batch")
    case = batch.cases[0]
    edge_offset = int(batch.graph.topology.graph_edge_ptr[0].item())
    node_offset = int(batch.graph.topology.graph_node_ptr[0].item())
    requirements = oracle.required_evidence
    requirement_rows: list[EvidenceRequirementObservation] = []
    candidate_rows: list[EvidenceCandidateObservation] = []

    for record in result.round_records:
        supervision = oracle.label(
            record.proposal,
            record.hypotheses,
            record.controller_state,
        )
        reachable = set(
            oracle.reachable_evidence_requirements(
                record.hypotheses,
                record.controller_state,
            )
        )
        outstanding = set(
            supervision.remaining_evidence_requirements
        )
        full = record.proposal.full_expansion
        scored = record.proposal.expansion
        logits = record.transition.refined_outputs.evidence_logits.detach()
        pre_context = (
            record.proposal.pre_context_outputs.evidence_logits.detach()
            if record.proposal.pre_context_outputs is not None
            else logits
        )
        labels = supervision.candidates.include_as_evidence
        selected_indices = record.actions.evidence_candidate_indices.to(
            torch.int64
        )
        frontier_indices = record.actions.frontier_candidate_indices.to(
            torch.int64
        )
        new_entries = record.transition.next_controller_state.evidence_ledger[
            len(record.controller_state.evidence_ledger) :
        ]
        selected_mask = torch.zeros(
            scored.total_arcs,
            dtype=torch.bool,
            device=scored.arc_ids.device,
        )
        frontier_mask = torch.zeros_like(selected_mask)
        if selected_indices.numel():
            selected_mask[selected_indices] = True
        if frontier_indices.numel():
            frontier_mask[frontier_indices] = True

        if logits.numel():
            order = torch.argsort(logits, descending=True, stable=True)
            ranks = torch.empty_like(order)
            ranks[order] = torch.arange(
                1,
                order.numel() + 1,
                dtype=torch.int64,
                device=order.device,
            )
            negative_logits = logits[~labels]
            best_negative = (
                float(negative_logits.max().item())
                if negative_logits.numel()
                else None
            )
        else:
            ranks = torch.empty(
                0,
                dtype=torch.int64,
                device=scored.arc_ids.device,
            )
            best_negative = None

        for requirement in requirements:
            enumerated_indices = _matching_indices(
                requirement,
                full.edge_ids,
                full.source_node_ids,
                full.destination_node_ids,
                edge_offset=edge_offset,
                node_offset=node_offset,
            )
            scored_indices = _matching_indices(
                requirement,
                scored.edge_ids,
                scored.source_node_ids,
                scored.destination_node_ids,
                edge_offset=edge_offset,
                node_offset=node_offset,
            )
            selected = bool(
                scored_indices.numel()
                and selected_mask[scored_indices].any().item()
            )
            frontier_selected = bool(
                scored_indices.numel()
                and frontier_mask[scored_indices].any().item()
            )
            recorded = any(
                _entry_matches(
                    requirement,
                    entry,
                    edge_offset=edge_offset,
                    node_offset=node_offset,
                )
                for entry in new_entries
            )
            if scored_indices.numel():
                requirement_logits = logits[scored_indices]
                best_local = int(
                    requirement_logits.argmax().item()
                )
                best_index = int(scored_indices[best_local].item())
                best_logit = float(logits[best_index].item())
                best_rank = int(ranks[best_index].item())
                margin = (
                    None
                    if best_negative is None
                    else best_logit - best_negative
                )
            else:
                best_logit = None
                best_rank = None
                margin = None
            requirement_rows.append(
                EvidenceRequirementObservation(
                    requirement=requirement,
                    round_index=record.controller_state.round_index,
                    outstanding_before=requirement in outstanding,
                    reachable=requirement in reachable,
                    enumerated=bool(enumerated_indices.numel()),
                    scored=bool(scored_indices.numel()),
                    selected=selected,
                    recorded=recorded,
                    frontier_selected=frontier_selected,
                    candidate_count=scored.total_arcs,
                    best_logit=best_logit,
                    best_rank=best_rank,
                    positive_negative_margin=margin,
                )
            )

        recorded_arcs = {entry.arc_id for entry in new_entries}
        for index in range(scored.total_arcs):
            candidate_rows.append(
                EvidenceCandidateObservation(
                    round_index=record.controller_state.round_index,
                    arc_id=int(scored.arc_ids[index].item()),
                    edge_id=int(scored.edge_ids[index].item()) - edge_offset,
                    source_node=(
                        int(scored.source_node_ids[index].item())
                        - node_offset
                    ),
                    destination_node=(
                        int(scored.destination_node_ids[index].item())
                        - node_offset
                    ),
                    logit=float(logits[index].item()),
                    pre_context_logit=float(pre_context[index].item()),
                    required=bool(labels[index].item()),
                    selected=bool(selected_mask[index].item()),
                    recorded=(
                        int(scored.arc_ids[index].item()) in recorded_arcs
                    ),
                    frontier_selected=bool(
                        frontier_mask[index].item()
                    ),
                )
            )

    edge_specific = any(
        requirement.edge_specific for requirement in requirements
    )
    required_keys = {
        (
            requirement.edge_id,
            requirement.source_node,
            requirement.destination_node,
        )
        for requirement in requirements
    }
    predicted_keys = _predicted_keys(
        batch,
        result.evidence_ledger,
        edge_specific=edge_specific,
    )
    true_positives = len(predicted_keys & required_keys)
    false_positives = len(predicted_keys - required_keys)
    false_negatives = len(required_keys - predicted_keys)
    exact = float(predicted_keys == required_keys)

    if candidate_rows:
        score_tensor = torch.tensor(
            [row.logit for row in candidate_rows],
            dtype=torch.float32,
        )
        label_tensor = torch.tensor(
            [row.required for row in candidate_rows],
            dtype=torch.bool,
        )
        average_precision = binary_average_precision(
            score_tensor,
            label_tensor,
        )
    else:
        average_precision = 0.0

    ranks_by_requirement: list[int] = []
    margins_by_requirement: list[float] = []
    for requirement in requirements:
        rows = [
            row
            for row in requirement_rows
            if row.requirement == requirement and row.outstanding_before
        ]
        observed_ranks = [
            row.best_rank for row in rows if row.best_rank is not None
        ]
        if observed_ranks:
            ranks_by_requirement.append(min(observed_ranks))
        elif rows:
            ranks_by_requirement.append(
                max(row.candidate_count for row in rows) + 1
            )
        observed_margins = [
            row.positive_negative_margin
            for row in rows
            if row.positive_negative_margin is not None
        ]
        if observed_margins:
            margins_by_requirement.append(max(observed_margins))

    return EvidencePipelineCaseReport(
        case_id=case.case_id,
        family=case.family.value,
        horizon=len(case.trace.rounds),
        requirements=requirements,
        requirement_observations=tuple(requirement_rows),
        candidate_observations=tuple(candidate_rows),
        exact_set_accuracy=exact,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        predicted_cardinality=len(predicted_keys),
        required_cardinality=len(required_keys),
        average_precision=average_precision,
        worst_positive_rank=(
            max(ranks_by_requirement) if ranks_by_requirement else None
        ),
        minimum_positive_negative_margin=(
            min(margins_by_requirement)
            if margins_by_requirement
            else None
        ),
    )


def _funnel_summary(
    observations: Iterable[EvidenceRequirementObservation],
) -> dict[str, int | float]:
    rows = [row for row in observations if row.outstanding_before]
    requirement_count = len(rows)
    reachable = sum(row.reachable for row in rows)
    enumerated = sum(row.enumerated for row in rows)
    scored = sum(row.scored for row in rows)
    selected = sum(row.selected for row in rows)
    recorded = sum(row.recorded for row in rows)
    frontier_selected = sum(row.frontier_selected for row in rows)
    selected_not_recorded = sum(
        row.selected and not row.recorded for row in rows
    )
    return {
        "requirement_opportunities": requirement_count,
        "reachable": reachable,
        "enumerated": enumerated,
        "scored": scored,
        "selected": selected,
        "recorded": recorded,
        "frontier_selected": frontier_selected,
        "selected_not_recorded": selected_not_recorded,
        "reachable_evidence_ceiling": (
            reachable / max(1, requirement_count)
        ),
        "scored_positive_coverage": scored / max(1, reachable),
        "selection_recall_conditioned_on_scored": (
            selected / max(1, scored)
        ),
        "recording_recall_conditioned_on_selected": (
            recorded / max(1, selected)
        ),
        "frontier_selection_recall": (
            frontier_selected / max(1, requirement_count)
        ),
    }


def _case_summary(
    reports: Sequence[EvidencePipelineCaseReport],
) -> dict[str, int | float | None]:
    required = sum(report.required_cardinality for report in reports)
    true_positives = sum(report.true_positives for report in reports)
    exact = sum(report.exact_set_accuracy for report in reports)
    false_positives = sum(report.false_positives for report in reports)
    false_negatives = sum(report.false_negatives for report in reports)
    positive_reports = [
        report for report in reports if report.required_cardinality > 0
    ]
    ranks = [
        report.worst_positive_rank
        for report in reports
        if report.worst_positive_rank is not None
    ]
    margins = [
        report.minimum_positive_negative_margin
        for report in reports
        if report.minimum_positive_negative_margin is not None
    ]
    return {
        "case_count": len(reports),
        "required_evidence_count": required,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "zero_positive_case_count": len(reports) - len(positive_reports),
        "exact_evidence_set_accuracy": exact / max(1, len(reports)),
        "false_positives_per_case": false_positives / max(1, len(reports)),
        "mean_predicted_cardinality": (
            sum(report.predicted_cardinality for report in reports)
            / max(1, len(reports))
        ),
        "mean_required_cardinality": (
            required / max(1, len(reports))
        ),
        "macro_evidence_average_precision": (
            sum(report.average_precision for report in positive_reports)
            / max(1, len(positive_reports))
        ),
        "mean_worst_positive_rank": (
            sum(ranks) / len(ranks) if ranks else None
        ),
        "minimum_positive_negative_margin": (
            min(margins) if margins else None
        ),
    }


def aggregate_evidence_pipeline(
    reports: Sequence[EvidencePipelineCaseReport],
    *,
    include_cases: bool = True,
) -> dict[str, object]:
    if not reports:
        raise ValueError("evidence pipeline aggregation requires case reports")

    def group_summary(
        selected: Sequence[EvidencePipelineCaseReport],
    ) -> dict[str, object]:
        first_observation_by_requirement: list[
            EvidenceRequirementObservation
        ] = []
        for report in selected:
            for requirement in report.requirements:
                rows = [
                    row
                    for row in report.requirement_observations
                    if row.requirement == requirement
                    and row.outstanding_before
                ]
                if not rows:
                    continue
                first = rows[0]
                first_observation_by_requirement.append(
                    EvidenceRequirementObservation(
                        requirement=requirement,
                        round_index=first.round_index,
                        outstanding_before=True,
                        reachable=any(row.reachable for row in rows),
                        enumerated=any(row.enumerated for row in rows),
                        scored=any(row.scored for row in rows),
                        selected=any(row.selected for row in rows),
                        recorded=any(row.recorded for row in rows),
                        frontier_selected=any(
                            row.frontier_selected for row in rows
                        ),
                        candidate_count=max(
                            row.candidate_count for row in rows
                        ),
                        best_logit=None,
                        best_rank=None,
                        positive_negative_margin=None,
                    )
                )
        return {
            **_funnel_summary(first_observation_by_requirement),
            **_case_summary(selected),
        }

    families = sorted({report.family for report in reports})
    horizons = sorted({report.horizon for report in reports})
    rounds = sorted(
        {
            row.round_index
            for report in reports
            for row in report.requirement_observations
        }
    )
    all_candidates = [
        row
        for report in reports
        for row in report.candidate_observations
    ]
    micro_ap = (
        binary_average_precision(
            torch.tensor(
                [row.logit for row in all_candidates],
                dtype=torch.float32,
            ),
            torch.tensor(
                [row.required for row in all_candidates],
                dtype=torch.bool,
            ),
        )
        if all_candidates
        else 0.0
    )
    result: dict[str, object] = {
        "overall": {
            **group_summary(reports),
            "micro_evidence_average_precision": micro_ap,
        },
        "by_family": {
            family: group_summary(
                [report for report in reports if report.family == family]
            )
            for family in families
        },
        "by_horizon": {
            str(horizon): group_summary(
                [report for report in reports if report.horizon == horizon]
            )
            for horizon in horizons
        },
        "by_round": {
            str(round_index): _funnel_summary(
                row
                for report in reports
                for row in report.requirement_observations
                if row.round_index == round_index
            )
            for round_index in rounds
        },
    }
    if include_cases:
        result["cases"] = [report.as_dict() for report in reports]
    return result
