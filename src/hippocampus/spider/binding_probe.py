from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F

from ..programs.renderer import SyntheticManifoldRenderer
from .evidence_readout import CanonicalBindingEvidenceReadout
from .model import CandidateScorerBase


_MODALITIES = ("query", "summary", "edge")
_PAIRS = tuple(combinations(_MODALITIES, 2))


@dataclass(frozen=True, slots=True)
class BindingRetrievalConfig:
    train_symbol_count: int = 1_024
    test_symbol_count: int = 256
    steps: int = 200
    batch_size: int = 256
    learning_rate: float = 0.01
    temperature: float = 0.07
    seed: int = 1701

    def __post_init__(self) -> None:
        for name in (
            "train_symbol_count",
            "test_symbol_count",
            "steps",
            "batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.test_symbol_count < 256:
            raise ValueError("test_symbol_count must be at least 256")
        if self.learning_rate <= 0 or self.temperature <= 0:
            raise ValueError("probe learning settings must be positive")


@dataclass(frozen=True, slots=True)
class BindingPairRetrieval:
    source: str
    target: str
    auroc: float
    top1_at_64: float
    top1_at_256: float


@dataclass(frozen=True, slots=True)
class BindingStageRetrieval:
    stage: str
    direct: tuple[BindingPairRetrieval, ...]
    fitted: tuple[BindingPairRetrieval, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "direct": [asdict(item) for item in self.direct],
            "fitted": [asdict(item) for item in self.fitted],
            "minimum_direct_auroc": min(item.auroc for item in self.direct),
            "minimum_direct_top1_at_64": min(
                item.top1_at_64 for item in self.direct
            ),
            "minimum_direct_top1_at_256": min(
                item.top1_at_256 for item in self.direct
            ),
            "minimum_fitted_auroc": min(item.auroc for item in self.fitted),
            "minimum_fitted_top1_at_64": min(
                item.top1_at_64 for item in self.fitted
            ),
            "minimum_fitted_top1_at_256": min(
                item.top1_at_256 for item in self.fitted
            ),
        }


@dataclass(frozen=True, slots=True)
class BindingRetrievalReport:
    config: BindingRetrievalConfig
    stages: tuple[BindingStageRetrieval, ...]
    canonical_gate_passed: bool
    diagnostic_only: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "stages": [stage.as_dict() for stage in self.stages],
            "canonical_gate_passed": self.canonical_gate_passed,
            "diagnostic_only": self.diagnostic_only,
        }


class _CrossModalProbe(nn.Module):
    def __init__(self, input_width: int) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                modality: nn.Linear(input_width, input_width)
                for modality in _MODALITIES
            }
        )

    def project(self, modality: str, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projections[modality](values), dim=-1)


def _symbol_values(
    renderer: SyntheticManifoldRenderer,
    symbols: tuple[str, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        modality: torch.stack(
            [
                renderer.render_symbol(symbol, modality=modality)
                for symbol in symbols
            ]
        ).to(device=device, dtype=dtype)
        for modality in _MODALITIES
    }


def _normalised(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), dim=-1)


def _auroc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    difference = positive[:, None] - negative[None, :]
    return float(
        (
            (difference > 0).float().mean()
            + 0.5 * (difference == 0).float().mean()
        ).item()
    )


def _top1(
    source: torch.Tensor,
    target: torch.Tensor,
    candidate_count: int,
) -> float:
    count = source.shape[0]
    offsets = torch.arange(candidate_count, device=source.device)
    candidate_ids = (
        torch.arange(count, device=source.device).unsqueeze(1)
        + offsets.unsqueeze(0)
    ) % count
    candidates = target[candidate_ids]
    scores = torch.einsum("nd,nkd->nk", source, candidates)
    return float((scores.argmax(dim=1) == 0).float().mean().item())


def _metrics(
    values: dict[str, torch.Tensor],
) -> tuple[BindingPairRetrieval, ...]:
    normalised = {name: _normalised(value) for name, value in values.items()}
    reports: list[BindingPairRetrieval] = []
    for source_name, target_name in _PAIRS:
        source = normalised[source_name]
        target = normalised[target_name]
        positive = (source * target).sum(dim=-1)
        negative = (source * target.roll(1, dims=0)).sum(dim=-1)
        reports.append(
            BindingPairRetrieval(
                source=source_name,
                target=target_name,
                auroc=_auroc(positive, negative),
                top1_at_64=_top1(source, target, 64),
                top1_at_256=_top1(source, target, 256),
            )
        )
    return tuple(reports)


def _fit_probe(
    train: dict[str, torch.Tensor],
    test: dict[str, torch.Tensor],
    *,
    config: BindingRetrievalConfig,
) -> tuple[BindingPairRetrieval, ...]:
    device = next(iter(train.values())).device
    input_width = int(next(iter(train.values())).shape[1])
    probe = _CrossModalProbe(input_width).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed + 1)
    for _ in range(config.steps):
        indices = torch.randperm(
            config.train_symbol_count,
            generator=generator,
            device=device,
        )[: min(config.batch_size, config.train_symbol_count)]
        target = torch.arange(indices.numel(), device=device)
        loss = torch.zeros((), device=device)
        for source_name, target_name in _PAIRS:
            source = probe.project(source_name, train[source_name][indices])
            destination = probe.project(
                target_name,
                train[target_name][indices],
            )
            logits = source @ destination.T / config.temperature
            loss = loss + 0.5 * (
                F.cross_entropy(logits, target)
                + F.cross_entropy(logits.T, target)
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        projected = {
            modality: probe.project(modality, values)
            for modality, values in test.items()
        }
    return _metrics(projected)


def run_model_binding_retrieval(
    model: CandidateScorerBase,
    renderer: SyntheticManifoldRenderer,
    *,
    config: BindingRetrievalConfig | None = None,
) -> BindingRetrievalReport:
    """Audit direct and linearly recoverable unseen-symbol alignment."""

    settings = config or BindingRetrievalConfig()
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    train_symbols = tuple(
        f"v07_probe_train_{settings.seed}_{index:06d}"
        for index in range(settings.train_symbol_count)
    )
    test_symbols = tuple(
        f"v07_probe_test_{settings.seed}_{index:06d}"
        for index in range(settings.test_symbol_count)
    )
    train_raw = _symbol_values(
        renderer,
        train_symbols,
        device=device,
        dtype=dtype,
    )
    test_raw = _symbol_values(
        renderer,
        test_symbols,
        device=device,
        dtype=dtype,
    )
    projections = {
        "query": model.query_projection,
        "summary": model.summary_projection,
        "edge": model.edge_projection,
    }
    was_training = model.training
    model.eval()
    with torch.no_grad():
        train_projected = {
            name: projections[name](values)
            for name, values in train_raw.items()
        }
        test_projected = {
            name: projections[name](values)
            for name, values in test_raw.items()
        }
    stage_values = [
        ("raw", train_raw, test_raw),
        ("family_projected", train_projected, test_projected),
    ]
    readout = model.evidence_readout
    if isinstance(readout, CanonicalBindingEvidenceReadout):
        canonicalizers = {
            "query": readout.query_canonicalizer,
            "summary": readout.summary_canonicalizer,
            "edge": readout.edge_canonicalizer,
        }
        with torch.no_grad():
            train_canonical = {
                name: canonicalizers[name](values)
                for name, values in train_projected.items()
            }
            test_canonical = {
                name: canonicalizers[name](values)
                for name, values in test_projected.items()
            }
        stage_values.append(
            ("evidence_canonical", train_canonical, test_canonical)
        )

    rng_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(settings.seed)
        stages = tuple(
            BindingStageRetrieval(
                stage=name,
                direct=_metrics(test_values),
                fitted=_fit_probe(
                    train_values,
                    test_values,
                    config=settings,
                ),
            )
            for name, train_values, test_values in stage_values
        )
    model.train(was_training)
    canonical = next(
        (stage for stage in stages if stage.stage == "evidence_canonical"),
        None,
    )
    canonical_gate_passed = bool(
        canonical is not None
        and min(item.auroc for item in canonical.direct) >= 0.99
        and min(item.top1_at_256 for item in canonical.direct) >= 0.95
    )
    return BindingRetrievalReport(
        config=settings,
        stages=stages,
        canonical_gate_passed=canonical_gate_passed,
    )
