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
