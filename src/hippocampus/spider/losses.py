from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import torch
from torch.nn import functional as F

from ..segmented import segment_logsumexp, segment_sum
from .types import CandidateOutputs
from .terminator import TerminationFactorTargets, TerminationOutput


EvidenceLossMode = Literal["plain", "balanced", "focal"]
EvidenceNullLossMode = Literal["plain", "graph_balanced"]


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
    evidence_ranking: float = 0.0
    evidence_plausible_ranking: float = 0.0
    evidence_ranking_margin: float = 0.2
    evidence_hard_negative_count: int = 4
    null_expansion: float = 1.0
    evidence_null: float = 0.0
    evidence_null_mode: EvidenceNullLossMode = "plain"
    evidence_null_margin: float = 0.0
    evidence_null_margin_value: float = 0.2
    evidence_null_hard_negative_count: int = 4
    evidence_cardinality: float = 0.0
    evidence_candidate_count: float = 0.0
    binding_alignment: float = 0.0
    binding_temperature: float = 0.07

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"{field.name} loss setting must be non-negative")
        if self.evidence_mode not in {"plain", "balanced", "focal"}:
            raise ValueError("evidence_mode must be plain, balanced, or focal")
        if self.evidence_null_mode not in {"plain", "graph_balanced"}:
            raise ValueError(
                "evidence_null_mode must be plain or graph_balanced"
            )
        if self.binding_temperature <= 0:
            raise ValueError("binding_temperature must be positive")
        if (
            self.evidence_null_margin > 0
            and self.evidence_null_hard_negative_count <= 0
        ):
            raise ValueError(
                "evidence NULL margin requires a hard-negative count"
            )
        if (
            self.evidence_ranking > 0
            or self.evidence_plausible_ranking > 0
        ) and self.evidence_hard_negative_count <= 0:
            raise ValueError(
                "ranked evidence loss requires a hard-negative count"
            )


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
    evidence_plausible_negative: torch.Tensor | None = None

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


def multi_positive_evidence_ranking_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    *,
    margin: float,
    max_negatives: int,
) -> tuple[torch.Tensor, int]:
    """Make every required item outrank the hardest invalid candidates."""

    if not (
        logits.shape == positive_mask.shape == negative_mask.shape
    ):
        raise ValueError("evidence ranking masks must align with logits")
    if margin < 0:
        raise ValueError("evidence ranking margin must be non-negative")
    if max_negatives <= 0:
        raise ValueError("max_negatives must be positive")
    positive_logits = logits.float()[positive_mask.bool()]
    negative_logits = logits.float()[negative_mask.bool()]
    if positive_logits.numel() == 0 or negative_logits.numel() == 0:
        return _zero(logits), 0
    hard_negatives = torch.topk(
        negative_logits,
        k=min(max_negatives, negative_logits.numel()),
        sorted=True,
    ).values
    pairwise = (
        hard_negatives.unsqueeze(0)
        - positive_logits.unsqueeze(1)
        + margin
    )
    zeros = pairwise.new_zeros((pairwise.shape[0], 1))
    loss = torch.logsumexp(torch.cat((zeros, pairwise), dim=1), dim=1).mean()
    return loss, int(positive_logits.numel())


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
        evidence_ranking, evidence_ranking_count = (
            multi_positive_evidence_ranking_loss(
                evidence_logits,
                supervision.include_as_evidence,
                ~supervision.include_as_evidence,
                margin=config.evidence_ranking_margin,
                max_negatives=config.evidence_hard_negative_count,
            )
        )
        plausible_mask = supervision.evidence_plausible_negative
        if plausible_mask is None:
            plausible_mask = torch.zeros_like(
                supervision.include_as_evidence
            )
        evidence_plausible_ranking, plausible_ranking_count = (
            multi_positive_evidence_ranking_loss(
                evidence_logits,
                supervision.include_as_evidence,
                plausible_mask,
                margin=config.evidence_ranking_margin,
                max_negatives=config.evidence_hard_negative_count,
            )
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
        evidence_ranking = _zero(outputs.evidence_logits)
        evidence_ranking_count = 0
        evidence_plausible_ranking = _zero(outputs.evidence_logits)
        plausible_ranking_count = 0
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
            "evidence_ranking": _term(
                evidence_ranking,
                weight=config.evidence_ranking,
                target_count=evidence_ranking_count,
            ),
            "evidence_plausible_ranking": _term(
                evidence_plausible_ranking,
                weight=config.evidence_plausible_ranking,
                target_count=plausible_ranking_count,
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
    candidate_parent_positions: torch.Tensor,
    *,
    config: SpiderLossConfig,
) -> LossTerm | None:
    """Supervise the explicit action that declines every candidate."""

    if null_logits is None:
        return None
    if not (
        acceptable.shape
        == depth_eligible.shape
        == candidate_parent_positions.shape
    ):
        raise ValueError("null-expansion supervision must align with candidates")
    targets = torch.ones_like(null_logits, dtype=torch.bool)
    preserving = acceptable.bool() & depth_eligible.bool()
    if bool(preserving.any().item()):
        parents = candidate_parent_positions[preserving].to(
            device=null_logits.device,
            dtype=torch.int64,
        )
        if bool((parents < 0).any().item()) or bool(
            (parents >= null_logits.numel()).any().item()
        ):
            raise IndexError("candidate parent position is out of range")
        targets[parents] = False
    raw = F.binary_cross_entropy_with_logits(
        null_logits.float(),
        targets.float(),
    )
    return _term(
        raw,
        weight=config.null_expansion,
        target_count=int(targets.numel()),
    )


def evidence_null_loss_term(
    null_logits: torch.Tensor | None,
    candidate_logits: torch.Tensor,
    candidate_targets: torch.Tensor,
    candidate_graph_ids: torch.Tensor,
    *,
    config: SpiderLossConfig,
) -> LossTerm | None:
    """Train a graph-local boundary between selected and rejected evidence."""

    if null_logits is None:
        return None
    if not (
        candidate_logits.shape
        == candidate_targets.shape
        == candidate_graph_ids.shape
    ):
        raise ValueError("evidence-null supervision must align with candidates")
    graph_ids = candidate_graph_ids.to(
        device=null_logits.device,
        dtype=torch.int64,
    )
    if graph_ids.numel() and (
        bool((graph_ids < 0).any().item())
        or bool((graph_ids >= null_logits.numel()).any().item())
    ):
        raise IndexError("candidate graph ID is out of range")
    target_count = int(candidate_targets.numel())
    if candidate_logits.numel() == 0:
        raw = _zero(null_logits)
    else:
        relative_logits = candidate_logits.float() - null_logits[graph_ids].float()
        elementwise = F.binary_cross_entropy_with_logits(
            relative_logits,
            candidate_targets.float(),
            reduction="none",
        )
        if config.evidence_null_mode == "plain":
            raw = elementwise.mean()
        else:
            positive = candidate_targets.bool()
            negative = ~positive
            graph_count = int(null_logits.numel())
            zeros = torch.zeros_like(elementwise)
            positive_sums = segment_sum(
                torch.where(positive, elementwise, zeros),
                row_owner_ids=graph_ids,
                num_segments=graph_count,
            ).values
            negative_sums = segment_sum(
                torch.where(negative, elementwise, zeros),
                row_owner_ids=graph_ids,
                num_segments=graph_count,
            ).values
            positive_counts = torch.bincount(
                graph_ids[positive],
                minlength=graph_count,
            )
            negative_counts = torch.bincount(
                graph_ids[negative],
                minlength=graph_count,
            )
            positive_valid = positive_counts > 0
            negative_valid = negative_counts > 0
            positive_means = positive_sums / positive_counts.clamp_min(1)
            negative_means = negative_sums / negative_counts.clamp_min(1)
            class_counts = (
                positive_valid.to(elementwise.dtype)
                + negative_valid.to(elementwise.dtype)
            )
            graph_losses = (
                positive_means * positive_valid
                + negative_means * negative_valid
            ) / class_counts.clamp_min(1)
            active_graphs = class_counts > 0
            raw = graph_losses[active_graphs].mean()
            target_count = int(active_graphs.sum().item())
    return _term(
        raw,
        weight=config.evidence_null,
        target_count=target_count,
    )


def evidence_null_margin_loss_term(
    null_logits: torch.Tensor | None,
    candidate_logits: torch.Tensor,
    candidate_targets: torch.Tensor,
    plausible_negative_mask: torch.Tensor | None,
    candidate_graph_ids: torch.Tensor,
    *,
    config: SpiderLossConfig,
) -> LossTerm | None:
    """Place positives and bounded graph-local hard negatives around NULL."""

    if null_logits is None or config.evidence_null_margin == 0:
        return None
    if not (
        candidate_logits.shape
        == candidate_targets.shape
        == candidate_graph_ids.shape
    ):
        raise ValueError("evidence NULL margin must align with candidates")
    if (
        plausible_negative_mask is not None
        and plausible_negative_mask.shape != candidate_logits.shape
    ):
        raise ValueError("plausible negatives must align with candidates")
    graph_ids = candidate_graph_ids.to(
        device=null_logits.device,
        dtype=torch.int64,
    )
    if graph_ids.numel() and (
        bool((graph_ids < 0).any().item())
        or bool((graph_ids >= null_logits.numel()).any().item())
    ):
        raise IndexError("candidate graph ID is out of range")
    if candidate_logits.numel() == 0:
        return _term(
            _zero(null_logits),
            weight=config.evidence_null_margin,
            target_count=0,
        )

    targets = candidate_targets.bool()
    plausible = (
        plausible_negative_mask.bool()
        if plausible_negative_mask is not None
        else torch.zeros_like(targets)
    )
    relative = candidate_logits.float() - null_logits[graph_ids].float()
    graph_losses: list[torch.Tensor] = []
    for graph_index in range(int(null_logits.numel())):
        graph_mask = graph_ids == graph_index
        positive_relative = relative[graph_mask & targets]
        negative_mask = graph_mask & ~targets
        plausible_mask = negative_mask & plausible
        selected_negative_mask = (
            plausible_mask
            if bool(plausible_mask.any().item())
            else negative_mask
        )
        negative_relative = relative[selected_negative_mask]
        if negative_relative.numel() > config.evidence_null_hard_negative_count:
            negative_relative = torch.topk(
                negative_relative,
                k=config.evidence_null_hard_negative_count,
                largest=True,
                sorted=False,
            ).values
        class_losses: list[torch.Tensor] = []
        if positive_relative.numel():
            class_losses.append(
                F.softplus(
                    config.evidence_null_margin_value - positive_relative
                ).mean()
            )
        if negative_relative.numel():
            class_losses.append(
                F.softplus(
                    config.evidence_null_margin_value + negative_relative
                ).mean()
            )
        if class_losses:
            graph_losses.append(torch.stack(class_losses).mean())
    raw = (
        torch.stack(graph_losses).mean()
        if graph_losses
        else _zero(null_logits)
    )
    return _term(
        raw,
        weight=config.evidence_null_margin,
        target_count=len(graph_losses),
    )


def evidence_cardinality_loss_term(
    cardinality_logits: torch.Tensor | None,
    required_cardinalities: torch.Tensor,
    *,
    config: SpiderLossConfig,
) -> LossTerm | None:
    """Supervise total required evidence count in classes 0, 1, 2, 3, 4+."""

    if cardinality_logits is None:
        return None
    if cardinality_logits.ndim != 2 or cardinality_logits.shape[1] != 5:
        raise ValueError("evidence cardinality logits must have shape [graphs, 5]")
    targets = required_cardinalities.to(
        device=cardinality_logits.device,
        dtype=torch.int64,
    )
    if targets.ndim != 1 or targets.numel() != cardinality_logits.shape[0]:
        raise ValueError("required cardinalities must align with graphs")
    targets = targets.clamp(min=0, max=4)
    if targets.numel() == 0:
        raw = _zero(cardinality_logits)
    else:
        raw = F.cross_entropy(cardinality_logits.float(), targets)
    return _term(
        raw,
        weight=config.evidence_cardinality,
        target_count=int(targets.numel()),
    )


def evidence_candidate_count_loss_term(
    count_logits: torch.Tensor | None,
    current_counts: torch.Tensor,
    *,
    config: SpiderLossConfig,
) -> LossTerm | None:
    """Supervise current-set evidence counts in classes 0, 1, 2, 3, 4+."""

    if count_logits is None:
        return None
    if count_logits.ndim != 2 or count_logits.shape[1] != 5:
        raise ValueError(
            "candidate evidence count logits must have shape [graphs, 5]"
        )
    targets = current_counts.to(device=count_logits.device, dtype=torch.int64)
    if targets.ndim != 1 or targets.numel() != count_logits.shape[0]:
        raise ValueError("candidate evidence counts must align with graphs")
    targets = targets.clamp(min=0, max=4)
    raw = (
        _zero(count_logits)
        if targets.numel() == 0
        else F.cross_entropy(count_logits.float(), targets)
    )
    return _term(
        raw,
        weight=config.evidence_candidate_count,
        target_count=int(targets.numel()),
    )


def termination_loss_report(
    output: TerminationOutput,
    targets: torch.Tensor,
    *,
    factor_targets: TerminationFactorTargets | None = None,
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
        if factor_targets is None:
            raise ValueError(
                "factorized termination requires direct state labels"
            )
        direct = factor_targets.validate(
            batch_size=int(resolved.numel()),
            device=output.logits.device,
        )
        sufficient_targets = direct.evidence_sufficient
        useful_targets = direct.useful_work_remaining
        sufficient_raw = F.binary_cross_entropy_with_logits(
            output.evidence_sufficient_logits.float(),
            sufficient_targets.float(),
        )
        useful_raw = F.binary_cross_entropy_with_logits(
            output.useful_work_remaining_logits.float(),
            useful_targets.float(),
        )
        answer_raw = F.binary_cross_entropy_with_logits(
            output.answer_supported_logits.float(),
            direct.answer_supported.float(),
        )
        answer_count = int(resolved.numel())
        unknown_mask = direct.unknown_mask
        if bool(unknown_mask.any().item()):
            unknown_raw = F.cross_entropy(
                output.unknown_logits[unknown_mask].float(),
                direct.unknown_reason[unknown_mask],
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
