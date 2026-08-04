from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import torch

from ..config import GraphSchema
from .schema import GraphProgramCase, ObservableAtom


RendererGeometry = Literal[
    "independent",
    "shared_additive",
    "orthogonal_aligned",
]


@dataclass(frozen=True, slots=True)
class RenderedCase:
    case_id: str
    query: torch.Tensor
    summaries: tuple[torch.Tensor, ...]
    contexts: tuple[torch.Tensor, ...]
    edges: tuple[torch.Tensor, ...]
    query_row_symbols: tuple[tuple[str, ...], ...] = ()
    summary_row_symbols: tuple[
        tuple[tuple[str, ...], ...], ...
    ] = ()
    context_row_symbols: tuple[
        tuple[tuple[str, ...], ...], ...
    ] = ()
    edge_row_symbols: tuple[
        tuple[tuple[str, ...], ...], ...
    ] = ()

    @property
    def query_dim(self) -> int:
        return int(self.query.shape[1])


class SyntheticManifoldRenderer:
    """Frozen deterministic renderer for unordered observable atoms."""

    renderer_version = "renderer-v0.4"

    def __init__(
        self,
        schema: GraphSchema,
        *,
        query_dim: int,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
        geometry: RendererGeometry = "independent",
        modality_embedding_scale: float = 0.25,
    ) -> None:
        if query_dim <= 0:
            raise ValueError("query_dim must be positive")
        if not dtype.is_floating_point:
            raise TypeError("renderer dtype must be floating point")
        if geometry not in {
            "independent",
            "shared_additive",
            "orthogonal_aligned",
        }:
            raise ValueError(f"unsupported renderer geometry {geometry!r}")
        if modality_embedding_scale < 0:
            raise ValueError("modality_embedding_scale must be non-negative")
        widths = {
            query_dim,
            schema.summary_dim,
            schema.context_dim,
            schema.edge_dim,
        }
        if geometry != "independent" and len(widths) != 1:
            raise ValueError(
                "aligned renderer geometries require one shared width"
            )
        self.schema = schema
        self.query_dim = query_dim
        self.seed = seed
        self.dtype = dtype
        self.geometry = geometry
        self.modality_embedding_scale = modality_embedding_scale
        self._symbol_cache: dict[tuple[str, str, int], torch.Tensor] = {}
        self._scalar_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._transform_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._modality_embedding_cache: dict[
            tuple[str, int], torch.Tensor
        ] = {}

    def _seed_for(self, *parts: object) -> int:
        payload = "|".join((str(self.seed), *(str(part) for part in parts)))
        digest = hashlib.sha256(payload.encode()).digest()
        return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)

    def _symbol_vector(
        self,
        symbol: str,
        *,
        family: str,
        width: int,
    ) -> torch.Tensor:
        key = (family, symbol, width)
        cached = self._symbol_cache.get(key)
        if cached is not None:
            return cached
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed_for("symbol", family, symbol, width))
        value = torch.randn(width, generator=generator, dtype=torch.float32)
        value = value / value.norm().clamp_min(1e-12)
        value = value.to(self.dtype)
        self._symbol_cache[key] = value
        return value

    def _scalar_direction(self, *, family: str, width: int) -> torch.Tensor:
        key = (family, width)
        cached = self._scalar_cache.get(key)
        if cached is not None:
            return cached
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed_for("scalar", family, width))
        direction = torch.randn(width, generator=generator, dtype=torch.float32)
        direction = direction / direction.norm().clamp_min(1e-12)
        direction = direction.to(self.dtype)
        self._scalar_cache[key] = direction
        return direction

    def _width_for_modality(self, modality: str) -> int:
        widths = {
            "query": self.query_dim,
            "summary": self.schema.summary_dim,
            "context": self.schema.context_dim,
            "edge": self.schema.edge_dim,
        }
        try:
            return widths[modality]
        except KeyError as exc:
            raise ValueError(f"unknown renderer modality {modality!r}") from exc

    def _shared_symbol_vector(self, symbol: str, width: int) -> torch.Tensor:
        return self._symbol_vector(
            symbol,
            family="shared-latent",
            width=width,
        )

    def _modality_embedding(self, modality: str, width: int) -> torch.Tensor:
        key = (modality, width)
        cached = self._modality_embedding_cache.get(key)
        if cached is not None:
            return cached
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self._seed_for("modality-embedding", modality, width)
        )
        value = torch.randn(width, generator=generator, dtype=torch.float32)
        value = (
            self.modality_embedding_scale
            * value
            / value.norm().clamp_min(1e-12)
        ).to(self.dtype)
        self._modality_embedding_cache[key] = value
        return value

    def _resolved_modality_transform(
        self,
        modality: str,
        width: int,
    ) -> torch.Tensor:
        key = (modality, width)
        cached = self._transform_cache.get(key)
        if cached is not None:
            return cached
        if self.geometry != "orthogonal_aligned":
            value = torch.eye(width, dtype=self.dtype)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                self._seed_for("modality-transform", modality, width)
            )
            matrix = torch.randn(
                (width, width),
                generator=generator,
                dtype=torch.float32,
            )
            value, triangular = torch.linalg.qr(matrix)
            signs = torch.sign(torch.diagonal(triangular))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            value = (value * signs.unsqueeze(0)).to(self.dtype)
        self._transform_cache[key] = value
        return value

    def modality_transform(self, modality: str) -> torch.Tensor:
        """Return a copy of the fixed transform used by one modality."""

        width = self._width_for_modality(modality)
        return self._resolved_modality_transform(modality, width).clone()

    def render_symbol(self, symbol: str, *, modality: str) -> torch.Tensor:
        """Render one opaque symbol without exposing its shared latent."""

        width = self._width_for_modality(modality)
        return self._render_atom(
            ObservableAtom((symbol,)),
            family=modality,
            width=width,
        )

    def _shared_scalar_value(self, scalar: float, width: int) -> torch.Tensor:
        direction = self._scalar_direction(
            family="shared-latent",
            width=width,
        )
        frequencies = torch.linspace(0.5, 2.0, width, dtype=self.dtype)
        continuous = (
            torch.sin(frequencies * scalar)
            + 0.5 * torch.cos(frequencies * scalar * 0.5)
        )
        return scalar * direction + 0.25 * continuous

    def _render_atom(
        self,
        atom: ObservableAtom,
        *,
        family: str,
        width: int,
    ) -> torch.Tensor:
        if self.geometry != "independent":
            latent = torch.zeros(width, dtype=self.dtype)
            if atom.symbols:
                for symbol in atom.symbols:
                    latent = latent + self._shared_symbol_vector(symbol, width)
                latent = latent / math.sqrt(len(atom.symbols))
            if atom.scalar is not None:
                latent = latent + self._shared_scalar_value(
                    float(atom.scalar),
                    width,
                )
            transformed = self._resolved_modality_transform(
                family,
                width,
            ) @ latent
            return transformed + self._modality_embedding(family, width)

        value = torch.zeros(width, dtype=self.dtype)
        if atom.symbols:
            for symbol in atom.symbols:
                value = value + self._symbol_vector(
                    symbol,
                    family=family,
                    width=width,
                )
            value = value / math.sqrt(len(atom.symbols))
        if atom.scalar is not None:
            scalar = float(atom.scalar)
            direction = self._scalar_direction(family=family, width=width)
            frequencies = torch.linspace(
                0.5,
                2.0,
                width,
                dtype=self.dtype,
            )
            continuous = (
                torch.sin(frequencies * scalar)
                + 0.5 * torch.cos(frequencies * scalar * 0.5)
            )
            value = value + scalar * direction + 0.25 * continuous
        return value

    def _render_rows(
        self,
        atoms: tuple[ObservableAtom, ...],
        *,
        family: str,
        width: int,
        permutation_seed: int,
        owner: int,
    ) -> torch.Tensor:
        rows, _ = self._render_rows_with_symbols(
            atoms,
            family=family,
            width=width,
            permutation_seed=permutation_seed,
            owner=owner,
        )
        return rows

    def _render_rows_with_symbols(
        self,
        atoms: tuple[ObservableAtom, ...],
        *,
        family: str,
        width: int,
        permutation_seed: int,
        owner: int,
    ) -> tuple[torch.Tensor, tuple[tuple[str, ...], ...]]:
        if atoms:
            rows = torch.stack(
                [
                    self._render_atom(atom, family=family, width=width)
                    for atom in atoms
                ]
            )
        else:
            rows = torch.empty((0, width), dtype=self.dtype)
        order = torch.arange(rows.shape[0])
        if rows.shape[0] > 1:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                self._seed_for(
                    "row-order",
                    permutation_seed,
                    family,
                    owner,
                )
            )
            order = torch.randperm(rows.shape[0], generator=generator)
            rows = rows[order]
        symbols = tuple(atoms[int(index)].symbols for index in order.tolist())
        return rows.contiguous(), symbols

    def render(
        self,
        case: GraphProgramCase,
        *,
        row_permutation_seed: int = 0,
    ) -> RenderedCase:
        query, query_symbols = self._render_rows_with_symbols(
            case.query_atoms,
            family="query",
            width=self.query_dim,
            permutation_seed=row_permutation_seed,
            owner=0,
        )
        rendered_summaries = tuple(
            self._render_rows_with_symbols(
                node.summary_atoms,
                family="summary",
                width=self.schema.summary_dim,
                permutation_seed=row_permutation_seed,
                owner=node_id,
            )
            for node_id, node in enumerate(case.nodes)
        )
        rendered_contexts = tuple(
            self._render_rows_with_symbols(
                node.context_atoms,
                family="context",
                width=self.schema.context_dim,
                permutation_seed=row_permutation_seed,
                owner=node_id,
            )
            for node_id, node in enumerate(case.nodes)
        )
        rendered_edges = tuple(
            self._render_rows_with_symbols(
                edge.atoms,
                family="edge",
                width=self.schema.edge_dim,
                permutation_seed=row_permutation_seed,
                owner=edge_id,
            )
            for edge_id, edge in enumerate(case.edges)
        )
        summaries = tuple(rows for rows, _ in rendered_summaries)
        contexts = tuple(rows for rows, _ in rendered_contexts)
        edges = tuple(rows for rows, _ in rendered_edges)
        return RenderedCase(
            case_id=case.case_id,
            query=query,
            summaries=summaries,
            contexts=contexts,
            edges=edges,
            query_row_symbols=query_symbols,
            summary_row_symbols=tuple(
                symbols for _, symbols in rendered_summaries
            ),
            context_row_symbols=tuple(
                symbols for _, symbols in rendered_contexts
            ),
            edge_row_symbols=tuple(
                symbols for _, symbols in rendered_edges
            ),
        )
