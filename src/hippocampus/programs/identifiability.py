from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F

from .renderer import SyntheticManifoldRenderer
from .schema import ObservableAtom


_MODALITIES = ("query", "summary", "edge")
_PAIRS = tuple(combinations(_MODALITIES, 2))


@dataclass(frozen=True, slots=True)
class IdentifiabilityProbeConfig:
    train_symbol_count: int = 4096
    test_symbol_count: int = 1024
    steps: int = 400
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
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True, slots=True)
class PairIdentifiability:
    source: str
    target: str
    auroc: float
    top1_at_64: float
    top1_at_256: float


@dataclass(frozen=True, slots=True)
class IdentifiabilityReport:
    renderer_geometry: str
    probe_config: IdentifiabilityProbeConfig
    pairs: tuple[PairIdentifiability, ...]
    macro_auroc: float
    macro_top1_at_64: float
    macro_top1_at_256: float
    minimum_auroc: float
    minimum_top1_at_64: float
    minimum_top1_at_256: float
    row_permutation_mismatches: int
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "renderer_geometry": self.renderer_geometry,
            "probe_config": asdict(self.probe_config),
            "pairs": [asdict(pair) for pair in self.pairs],
            "macro_auroc": self.macro_auroc,
            "macro_top1_at_64": self.macro_top1_at_64,
            "macro_top1_at_256": self.macro_top1_at_256,
            "minimum_auroc": self.minimum_auroc,
            "minimum_top1_at_64": self.minimum_top1_at_64,
            "minimum_top1_at_256": self.minimum_top1_at_256,
            "row_permutation_mismatches": self.row_permutation_mismatches,
            "passed": self.passed,
        }


class _TwoProjectionProbe(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {name: nn.Linear(width, width) for name in _MODALITIES}
        )

    def project(self, modality: str, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projections[modality](values), dim=-1)


def _symbol_matrix(
    renderer: SyntheticManifoldRenderer,
    symbols: tuple[str, ...],
    modality: str,
) -> torch.Tensor:
    return torch.stack(
        [renderer.render_symbol(symbol, modality=modality) for symbol in symbols]
    ).float()


def _auroc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (
            (comparisons > 0).float().mean()
            + 0.5 * (comparisons == 0).float().mean()
        ).item()
    )


def _top1(
    source: torch.Tensor,
    target: torch.Tensor,
    distractor_count: int,
) -> float:
    count = source.shape[0]
    offsets = torch.arange(distractor_count)
    candidate_ids = (
        torch.arange(count).unsqueeze(1) + offsets.unsqueeze(0)
    ) % count
    candidates = target[candidate_ids]
    scores = torch.einsum("nd,nkd->nk", source, candidates)
    return float((scores.argmax(dim=1) == 0).float().mean().item())


def _row_permutation_mismatches(
    renderer: SyntheticManifoldRenderer,
    symbols: tuple[str, ...],
) -> int:
    atoms = tuple(ObservableAtom((symbol,)) for symbol in symbols[:16])
    first = renderer._render_rows(
        atoms,
        family="query",
        width=renderer.query_dim,
        permutation_seed=11,
        owner=0,
    )
    second = renderer._render_rows(
        atoms,
        family="query",
        width=renderer.query_dim,
        permutation_seed=29,
        owner=0,
    )
    distances = torch.cdist(first.float(), second.float())
    mismatch = bool(
        (distances.amin(dim=0) > 1e-6).any()
        or (distances.amin(dim=1) > 1e-6).any()
    )
    return int(mismatch)


def run_renderer_identifiability(
    renderer: SyntheticManifoldRenderer,
    *,
    config: IdentifiabilityProbeConfig | None = None,
) -> IdentifiabilityReport:
    """Fit a linear cross-modal probe and evaluate only unseen symbols."""

    settings = config or IdentifiabilityProbeConfig()
    widths = {
        renderer.query_dim,
        renderer.schema.summary_dim,
        renderer.schema.edge_dim,
    }
    if len(widths) != 1:
        raise ValueError("identifiability probe requires one shared width")
    width = widths.pop()
    train_symbols = tuple(
        f"probe_train_{settings.seed}_{index:06d}"
        for index in range(settings.train_symbol_count)
    )
    test_symbols = tuple(
        f"probe_test_{settings.seed}_{index:06d}"
        for index in range(settings.test_symbol_count)
    )
    train_values = {
        modality: _symbol_matrix(renderer, train_symbols, modality)
        for modality in _MODALITIES
    }
    test_values = {
        modality: _symbol_matrix(renderer, test_symbols, modality)
        for modality in _MODALITIES
    }

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings.seed)
        probe = _TwoProjectionProbe(width)
        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=settings.learning_rate,
            weight_decay=1e-4,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(settings.seed + 1)
        for _ in range(settings.steps):
            indices = torch.randperm(
                settings.train_symbol_count,
                generator=generator,
            )[: min(settings.batch_size, settings.train_symbol_count)]
            target = torch.arange(indices.numel())
            loss = torch.zeros(())
            for source_modality, target_modality in _PAIRS:
                source = probe.project(
                    source_modality,
                    train_values[source_modality][indices],
                )
                destination = probe.project(
                    target_modality,
                    train_values[target_modality][indices],
                )
                logits = source @ destination.T / settings.temperature
                loss = loss + 0.5 * (
                    F.cross_entropy(logits, target)
                    + F.cross_entropy(logits.T, target)
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            projected = {
                modality: probe.project(modality, values)
                for modality, values in test_values.items()
            }
            pair_reports: list[PairIdentifiability] = []
            for source_modality, target_modality in _PAIRS:
                source = projected[source_modality]
                target = projected[target_modality]
                positive = (source * target).sum(dim=-1)
                negative = (source * target.roll(1, dims=0)).sum(dim=-1)
                pair_reports.append(
                    PairIdentifiability(
                        source=source_modality,
                        target=target_modality,
                        auroc=_auroc(positive, negative),
                        top1_at_64=_top1(source, target, 64),
                        top1_at_256=_top1(source, target, 256),
                    )
                )

    pairs = tuple(pair_reports)
    macro_auroc = sum(pair.auroc for pair in pairs) / len(pairs)
    macro_64 = sum(pair.top1_at_64 for pair in pairs) / len(pairs)
    macro_256 = sum(pair.top1_at_256 for pair in pairs) / len(pairs)
    minimum_auroc = min(pair.auroc for pair in pairs)
    minimum_64 = min(pair.top1_at_64 for pair in pairs)
    minimum_256 = min(pair.top1_at_256 for pair in pairs)
    row_mismatches = _row_permutation_mismatches(renderer, test_symbols)
    passed = (
        minimum_auroc >= 0.99
        and minimum_64 >= 0.95
        and minimum_256 >= 0.85
        and row_mismatches == 0
    )
    return IdentifiabilityReport(
        renderer_geometry=renderer.geometry,
        probe_config=settings,
        pairs=pairs,
        macro_auroc=macro_auroc,
        macro_top1_at_64=macro_64,
        macro_top1_at_256=macro_256,
        minimum_auroc=minimum_auroc,
        minimum_top1_at_64=minimum_64,
        minimum_top1_at_256=minimum_256,
        row_permutation_mismatches=row_mismatches,
        passed=passed,
    )
