from __future__ import annotations

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F


TERMINATION_CLASS_COUNT = 6


@dataclass(frozen=True, slots=True)
class TerminationOutput:
    logits: torch.Tensor
    stop_logits: torch.Tensor | None = None
    answer_logits: torch.Tensor | None = None
    unknown_logits: torch.Tensor | None = None
    evidence_sufficient_logits: torch.Tensor | None = None
    useful_work_remaining_logits: torch.Tensor | None = None
    answer_supported_logits: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class TerminationFactorTargets:
    """Direct exact-state labels for the factorized termination heads."""

    evidence_sufficient: torch.Tensor
    useful_work_remaining: torch.Tensor
    answer_supported: torch.Tensor
    unknown_reason: torch.Tensor
    unknown_mask: torch.Tensor

    def validate(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> "TerminationFactorTargets":
        boolean_names = (
            "evidence_sufficient",
            "useful_work_remaining",
            "answer_supported",
            "unknown_mask",
        )
        for name in boolean_names:
            value = getattr(self, name)
            if value.shape != (batch_size,) or value.dtype != torch.bool:
                raise ValueError(
                    f"{name} must be bool[{batch_size}]"
                )
            if value.device != device:
                raise ValueError(f"{name} must share the output device")
        if (
            self.unknown_reason.shape != (batch_size,)
            or self.unknown_reason.dtype != torch.int64
        ):
            raise ValueError(
                f"unknown_reason must be int64[{batch_size}]"
            )
        if self.unknown_reason.device != device:
            raise ValueError("unknown_reason must share the output device")
        if bool(
            (
                (self.unknown_reason[self.unknown_mask] < 0)
                | (self.unknown_reason[self.unknown_mask] >= 4)
            ).any().item()
        ):
            raise ValueError("unknown reasons must be in [0, 4)")
        return self


class TerminationHead(nn.Module):
    def __init__(self, d_model: int, control_width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(3 * d_model + control_width),
            nn.Linear(3 * d_model + control_width, d_model),
            nn.GELU(),
            nn.Linear(d_model, TERMINATION_CLASS_COUNT),
        )

    def forward(
        self,
        query: torch.Tensor,
        evidence: torch.Tensor,
        frontier: torch.Tensor,
        control: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat((query, evidence, frontier, control), dim=-1))


class HierarchicalTerminationHead(nn.Module):
    """CONTINUE/STOP, ANSWER/UNKNOWN, then four unknown reasons."""

    def __init__(self, d_model: int, control_width: int) -> None:
        super().__init__()
        width = 3 * d_model + control_width
        self.trunk = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, d_model),
            nn.GELU(),
        )
        self.stop = nn.Linear(d_model, 1)
        self.answer = nn.Linear(d_model, 1)
        self.unknown = nn.Linear(d_model, 4)

    def forward(
        self,
        query: torch.Tensor,
        evidence: torch.Tensor,
        frontier: torch.Tensor,
        control: torch.Tensor,
    ) -> TerminationOutput:
        hidden = self.trunk(
            torch.cat((query, evidence, frontier, control), dim=-1)
        )
        stop = self.stop(hidden).squeeze(-1)
        answer = self.answer(hidden).squeeze(-1)
        unknown = self.unknown(hidden)
        continue_log_probability = F.logsigmoid(-stop)
        answer_log_probability = F.logsigmoid(stop) + F.logsigmoid(answer)
        unknown_base = F.logsigmoid(stop) + F.logsigmoid(-answer)
        unknown_log_probability = (
            unknown_base[:, None] + F.log_softmax(unknown, dim=-1)
        )
        logits = torch.cat(
            (
                continue_log_probability[:, None],
                answer_log_probability[:, None],
                unknown_log_probability,
            ),
            dim=-1,
        )
        return TerminationOutput(
            logits=logits,
            stop_logits=stop,
            answer_logits=answer,
            unknown_logits=unknown,
        )


class FactorizedTerminationHead(nn.Module):
    """Evidence sufficiency, reachable work, answer support, and reason."""

    def __init__(self, d_model: int, control_width: int) -> None:
        super().__init__()
        width = 3 * d_model + control_width
        self.trunk = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, d_model),
            nn.GELU(),
        )
        self.evidence_sufficient = nn.Linear(d_model, 1)
        self.useful_work_remaining = nn.Linear(d_model, 1)
        self.answer_supported = nn.Linear(d_model, 1)
        self.unknown_reason = nn.Linear(d_model, 4)

    def forward(
        self,
        query: torch.Tensor,
        evidence: torch.Tensor,
        frontier: torch.Tensor,
        control: torch.Tensor,
    ) -> TerminationOutput:
        hidden = self.trunk(
            torch.cat((query, evidence, frontier, control), dim=-1)
        )
        sufficient = self.evidence_sufficient(hidden).squeeze(-1)
        useful = self.useful_work_remaining(hidden).squeeze(-1)
        answer = self.answer_supported(hidden).squeeze(-1)
        unknown = self.unknown_reason(hidden)

        continue_log_probability = (
            F.logsigmoid(-sufficient) + F.logsigmoid(useful)
        )
        answer_log_probability = (
            F.logsigmoid(sufficient) + F.logsigmoid(answer)
        )
        unknown_log_probability = torch.logaddexp(
            F.logsigmoid(sufficient) + F.logsigmoid(-answer),
            F.logsigmoid(-sufficient) + F.logsigmoid(-useful),
        )
        logits = torch.cat(
            (
                continue_log_probability[:, None],
                answer_log_probability[:, None],
                unknown_log_probability[:, None]
                + F.log_softmax(unknown, dim=-1),
            ),
            dim=-1,
        )
        return TerminationOutput(
            logits=logits,
            unknown_logits=unknown,
            evidence_sufficient_logits=sufficient,
            useful_work_remaining_logits=useful,
            answer_supported_logits=answer,
        )
