from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import torch
from torch.nn import functional as F

from ..segmented import segment_logsumexp, segment_sum
from .types import CandidateOutputs
from .terminator import TerminationOutput


EvidenceLossMode = Literal["plain", "balanced", "focal"]


@dataclass(frozen=True, slots=True)
class SpiderLossConfig:
    priority: float = 1.0
    expand: float = 1.0
    context: float = 0.5
    evidence: float = 0.75
    support_conflict: float = 0.5
    remaining_cost: float = 0.25
    termination: float = 1.0
    consistency: float = 0.0
    search_cost: float = 0.0
    context_cost: float = 0.0
    evidence_set: float = 0.0
    evidence_mode: EvidenceLossMode = "plain"
    evidence_positive_weight: float | None = None
    evidence_focal_gamma: float = 2.0
    evidence_false_positive_penalty: float = 0.1
    null_expansion: float = 1.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"{field.name} loss setting must be non-negative")
        if self.evidence_mode not in {"plain", "balanced", "focal"}:
            raise ValueError("evidence_mode must be plain, balanced, or focal")


@dataclass(frozen=True, slots=True)
class LossTerm:
    raw: torch.Tensor
    weighted: torch.Tensor
    target_count: int


@dataclass(frozen=True, slots=True)
class CandidateSupervision:
    acceptable: torch.Tensor
    context_has_value: torch.Tensor
    include_as_evidence: torch.Tensor
    remaining_cost: torch.Tensor
    support: torch.Tensor
    conflict: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.acceptable.numel())


@dataclass(frozen=True, slots=True)
class SpiderLossReport:
    terms: dict[str, LossTerm]

    @property
    def total(self) -> torch.Tensor:
        if not self.terms:
            raise ValueError("a loss report requires at least one term")
        return torch.stack(
            [term.weighted for term in self.terms.values()]
        ).sum()

    def detached_metrics(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {}
        for name, term in self.terms.items():
            metrics[f"loss/{name}/raw"] = float(term.raw.detach().item())
            metrics[f"loss/{name}/weighted"] = float(
                term.weighted.detach().item()
            )
            metrics[f"loss/{name}/targets"] = term.target_count
        metrics["loss/total"] = float(self.total.detach().item())
        return metrics


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _term(
    raw: torch.Tensor,
    *,
    weight: float,
    target_count: int,
) -> LossTerm:
    return LossTerm(raw=raw, weighted=raw * weight, target_count=target_count)


def multi_positive_priority_loss(
    logits: torch.Tensor,
    acceptable: torch.Tensor,
    frontier_positions: torch.Tensor,
    *,
    frontier_count: int,
) -> tuple[torch.Tensor, int]:
    if logits.numel() == 0:
        return _zero(logits), 0
    owners = frontier_positions.to(torch.int64)
    all_lse = segment_logsumexp(
        logits.float().unsqueeze(-1),
        row_owner_ids=owners,
        num_segments=frontier_count,
        output_dtype=torch.float32,
    )
    masked = torch.where(
        acceptable,
        logits.float(),
        torch.full_like(logits.float(), -1e9),
    )
    positive_lse = segment_logsumexp(
        masked.unsqueeze(-1),
        row_owner_ids=owners,
        num_segments=frontier_count,
        output_dtype=torch.float32,
    )
    positive_counts = segment_sum(
        acceptable.to(logits.dtype).unsqueeze(-1),
        row_owner_ids=owners,
        num_segments=frontier_count,
        output_dtype=torch.float32,
    ).values.squeeze(-1)
    valid = positive_counts > 0
    count = int(valid.sum().item())
    if count == 0:
        return _zero(logits), 0
    loss = (
        all_lse.values.squeeze(-1)[valid]
        - positive_lse.values.squeeze(-1)[valid]
    ).mean()
    return loss, count


def candidate_loss_report(
    outputs: CandidateOutputs,
    supervision: CandidateSupervision,
    frontier_positions: torch.Tensor,
    *,
    frontier_count: int,
    config: SpiderLossConfig,
    context_logits: torch.Tensor | None = None,
) -> SpiderLossReport:
    if supervision.count != outputs.candidate_count:
        raise ValueError("candidate supervision must align with model outputs")
    count = supervision.count
    priority, priority_count = multi_positive_priority_loss(
        outputs.priority_logits,
        supervision.acceptable,
        frontier_positions,
        frontier_count=frontier_count,
    )
    if count:
        expand = F.binary_cross_entropy_with_logits(
            outputs.expand_logits.float(),
            supervision.acceptable.float(),
        )
        context_source = (
            outputs.context_logits
            if context_logits is None
            else context_logits
        )
        context = F.binary_cross_entropy_with_logits(
            context_source.float(),
            supervision.context_has_value.float(),
        )
        evidence_targets = supervision.include_as_evidence.float()
        evidence_logits = outputs.evidence_logits.float()
        positive_count = int(
            supervision.include_as_evidence.sum().item()
        )
        negative_count = count - positive_count
        if config.evidence_mode == "plain":
            evidence = F.binary_cross_entropy_with_logits(
                evidence_logits,
                evidence_targets,
            )
        elif config.evidence_mode == "balanced":
            positive_weight = (
                config.evidence_positive_weight
                if config.evidence_positive_weight is not None
                else negative_count / max(1, positive_count)
            )
            evidence = F.binary_cross_entropy_with_logits(
                evidence_logits,
                evidence_targets,
                pos_weight=evidence_logits.new_tensor(
                    max(1.0, positive_weight)
                ),
            )
        else:
            elementwise = F.binary_cross_entropy_with_logits(
                evidence_logits,
                evidence_targets,
                reduction="none",
            )
            probabilities = torch.sigmoid(evidence_logits)
            target_probability = torch.where(
                supervision.include_as_evidence,
                probabilities,
                1.0 - probabilities,
            )
            evidence = (
                (1.0 - target_probability).pow(
                    config.evidence_focal_gamma
                )
                * elementwise
            ).mean()
        probabilities = torch.sigmoid(evidence_logits)
        if positive_count:
            positive_logits = evidence_logits[
                supervision.include_as_evidence
            ]
            positive_mass = (
                torch.logsumexp(evidence_logits, dim=0)
                - torch.logsumexp(positive_logits, dim=0)
            )
            soft_recall = 1.0 - probabilities[
                supervision.include_as_evidence
            ].mean()
        else:
            positive_mass = _zero(evidence_logits)
            soft_recall = _zero(evidence_logits)
        false_positive = (
            probabilities[~supervision.include_as_evidence].mean()
            if negative_count
            else _zero(evidence_logits)
        )
        evidence_set = (
            0.5 * positive_mass
            + 0.5 * soft_recall
            + config.evidence_false_positive_penalty * false_positive
        )
        support = F.binary_cross_entropy_with_logits(
            outputs.support_logits.float(),
            supervision.support.float(),
        )
        conflict = F.binary_cross_entropy_with_logits(
            outputs.conflict_logits.float(),
            supervision.conflict.float(),
        )
        support_conflict = 0.5 * (support + conflict)
        remaining = F.smooth_l1_loss(
            outputs.remaining_cost.float(),
            supervision.remaining_cost.float(),
        )
        search_cost = torch.sigmoid(outputs.expand_logits.float()).mean()
        context_cost = torch.sigmoid(context_source.float()).mean()
    else:
        expand = _zero(outputs.expand_logits)
        context = _zero(outputs.context_logits)
        evidence = _zero(outputs.evidence_logits)
        evidence_set = _zero(outputs.evidence_logits)
        support_conflict = _zero(outputs.support_logits)
        remaining = _zero(outputs.remaining_cost)
        search_cost = _zero(outputs.expand_logits)
        context_cost = _zero(outputs.context_logits)
    return SpiderLossReport(
        terms={
            "priority": _term(
                priority,
                weight=config.priority,
                target_count=priority_count,
            ),
            "expand": _term(expand, weight=config.expand, target_count=count),
            "context": _term(context, weight=config.context, target_count=count),
            "evidence": _term(
                evidence,
                weight=config.evidence,
                target_count=count,
            ),
            "evidence_set": _term(
                evidence_set,
                weight=config.evidence_set,
                target_count=(
                    int(supervision.include_as_evidence.sum().item())
                    if count
                    else 0
                ),
            ),
            "support_conflict": _term(
                support_conflict,
                weight=config.support_conflict,
                target_count=2 * count,
            ),
            "remaining_cost": _term(
                remaining,
                weight=config.remaining_cost,
                target_count=count,
            ),
            "search_cost": _term(
                search_cost,
                weight=config.search_cost,
                target_count=count,
            ),
            "context_cost": _term(
                context_cost,
                weight=config.context_cost,
                target_count=count,
            ),
        }
    )


def termination_loss_term(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    config: SpiderLossConfig,
) -> LossTerm:
    if targets.numel() == 0:
        raw = _zero(logits)
    else:
        raw = F.cross_entropy(logits.float(), targets.to(torch.int64))
    return _term(
        raw,
        weight=config.termination,
        target_count=int(targets.numel()),
    )


def null_expansion_loss_term(
    null_logits: torch.Tensor | None,
    acceptable: torch.Tensor,
    depth_eligible: torch.Tensor,
    candidate_graph_ids: torch.Tensor,
    *,
    config: SpiderLossConfig,
) -> LossTerm | None:
    """Supervise the explicit action that declines every candidate."""

    if null_logits is None:
        return None
    if not (
        acceptable.shape
        == depth_eligible.shape
        == candidate_graph_ids.shape
    ):
        raise ValueError("null-expansion supervision must align with candidates")
    targets = torch.ones_like(null_logits, dtype=torch.bool)
    preserving = acceptable.bool() & depth_eligible.bool()
    if bool(preserving.any().item()):
        targets[
            candidate_graph_ids[preserving].to(
                device=null_logits.device,
                dtype=torch.int64,
            )
        ] = False
    raw = F.binary_cross_entropy_with_logits(
        null_logits.float(),
        targets.float(),
    )
    return _term(
        raw,
        weight=config.null_expansion,
        target_count=int(targets.numel()),
    )


def termination_loss_report(
    output: TerminationOutput,
    targets: torch.Tensor,
    *,
    config: SpiderLossConfig,
) -> SpiderLossReport:
    """Return flat or masked hierarchical termination objectives."""

    resolved = targets.to(device=output.logits.device, dtype=torch.int64)
    if output.evidence_sufficient_logits is not None:
        if (
            output.useful_work_remaining_logits is None
            or output.answer_supported_logits is None
            or output.unknown_logits is None
        ):
            raise ValueError("factorized termination output is incomplete")
        sufficient_targets = (resolved == 1) | (resolved == 3)
        useful_targets = resolved == 0
        sufficient_raw = F.binary_cross_entropy_with_logits(
            output.evidence_sufficient_logits.float(),
            sufficient_targets.float(),
        )
        useful_raw = F.binary_cross_entropy_with_logits(
            output.useful_work_remaining_logits.float(),
            useful_targets.float(),
        )
        if bool(sufficient_targets.any().item()):
            answer_raw = F.binary_cross_entropy_with_logits(
                output.answer_supported_logits[sufficient_targets].float(),
                (resolved[sufficient_targets] == 1).float(),
            )
            answer_count = int(sufficient_targets.sum().item())
        else:
            answer_raw = _zero(output.answer_supported_logits)
            answer_count = 0
        unknown_mask = resolved >= 2
        if bool(unknown_mask.any().item()):
            unknown_raw = F.cross_entropy(
                output.unknown_logits[unknown_mask].float(),
                resolved[unknown_mask] - 2,
            )
            unknown_count = int(unknown_mask.sum().item())
        else:
            unknown_raw = _zero(output.unknown_logits)
            unknown_count = 0
        return SpiderLossReport(
            {
                "termination_evidence_sufficient": _term(
                    sufficient_raw,
                    weight=config.termination,
                    target_count=int(resolved.numel()),
                ),
                "termination_useful_work": _term(
                    useful_raw,
                    weight=config.termination,
                    target_count=int(resolved.numel()),
                ),
                "termination_answer_supported": _term(
                    answer_raw,
                    weight=config.termination,
                    target_count=answer_count,
                ),
                "termination_unknown": _term(
                    unknown_raw,
                    weight=config.termination,
                    target_count=unknown_count,
                ),
            }
        )
    if output.stop_logits is None:
        return SpiderLossReport(
            {
                "termination": termination_loss_term(
                    output.logits,
                    resolved,
                    config=config,
                )
            }
        )
    if (
        output.answer_logits is None
        or output.unknown_logits is None
    ):
        raise ValueError("hierarchical termination output is incomplete")
    stop_targets = resolved != 0
    stop_raw = F.binary_cross_entropy_with_logits(
        output.stop_logits.float(),
        stop_targets.float(),
    )
    stop_term = _term(
        stop_raw,
        weight=config.termination,
        target_count=int(resolved.numel()),
    )
    if bool(stop_targets.any().item()):
        answer_targets = resolved[stop_targets] == 1
        answer_raw = F.binary_cross_entropy_with_logits(
            output.answer_logits[stop_targets].float(),
            answer_targets.float(),
        )
        answer_count = int(stop_targets.sum().item())
    else:
        answer_raw = _zero(output.answer_logits)
        answer_count = 0
    unknown_mask = resolved >= 2
    if bool(unknown_mask.any().item()):
        unknown_raw = F.cross_entropy(
            output.unknown_logits[unknown_mask].float(),
            resolved[unknown_mask] - 2,
        )
        unknown_count = int(unknown_mask.sum().item())
    else:
        unknown_raw = _zero(output.unknown_logits)
        unknown_count = 0
    return SpiderLossReport(
        {
            "termination_stop": stop_term,
            "termination_answer": _term(
                answer_raw,
                weight=config.termination,
                target_count=answer_count,
            ),
            "termination_unknown": _term(
                unknown_raw,
                weight=config.termination,
                target_count=unknown_count,
            ),
        }
    )


def behavioural_consistency_loss(
    first: CandidateOutputs,
    second: CandidateOutputs,
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
    *,
    weight: float = 1.0,
) -> LossTerm:
    """Compare aligned behaviours without forcing hidden-coordinate equality."""

    left = first_indices.to(
        device=first.priority_logits.device,
        dtype=torch.int64,
    )
    right = second_indices.to(
        device=second.priority_logits.device,
        dtype=torch.int64,
    )
    if left.numel() != right.numel():
        raise ValueError("behavioural alignments must have equal lengths")
    count = int(left.numel())
    if count == 0:
        raw = _zero(first.priority_logits) + _zero(second.priority_logits)
        return _term(raw, weight=weight, target_count=0)
    first_priority = torch.log_softmax(
        first.priority_logits[left].float(),
        dim=0,
    )
    second_priority = torch.log_softmax(
        second.priority_logits[right].float(),
        dim=0,
    )
    priority = 0.5 * (
        F.kl_div(
            first_priority,
            second_priority.exp(),
            reduction="batchmean",
        )
        + F.kl_div(
            second_priority,
            first_priority.exp(),
            reduction="batchmean",
        )
    )
    action_pairs = (
        (first.expand_logits, second.expand_logits),
        (first.context_logits, second.context_logits),
        (first.evidence_logits, second.evidence_logits),
        (first.support_logits, second.support_logits),
        (first.conflict_logits, second.conflict_logits),
    )
    actions = torch.stack(
        [
            F.mse_loss(
                torch.sigmoid(first_logits[left].float()),
                torch.sigmoid(second_logits[right].float()),
            )
            for first_logits, second_logits in action_pairs
        ]
    ).mean()
    raw = priority + actions
    return _term(raw, weight=weight, target_count=count)
