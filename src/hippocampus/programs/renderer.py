from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from ..config import GraphSchema
from .schema import GraphProgramCase, ObservableAtom


@dataclass(frozen=True, slots=True)
class RenderedCase:
    case_id: str
    query: torch.Tensor
    summaries: tuple[torch.Tensor, ...]
    contexts: tuple[torch.Tensor, ...]
    edges: tuple[torch.Tensor, ...]

    @property
    def query_dim(self) -> int:
        return int(self.query.shape[1])


class SyntheticManifoldRenderer:
    """Frozen deterministic renderer for unordered observable atoms."""

    def __init__(
        self,
        schema: GraphSchema,
        *,
        query_dim: int,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if query_dim <= 0:
            raise ValueError("query_dim must be positive")
        if not dtype.is_floating_point:
            raise TypeError("renderer dtype must be floating point")
        self.schema = schema
        self.query_dim = query_dim
        self.seed = seed
        self.dtype = dtype
        self._symbol_cache: dict[tuple[str, str, int], torch.Tensor] = {}
        self._scalar_cache: dict[tuple[str, int], torch.Tensor] = {}

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

    def _render_atom(
        self,
        atom: ObservableAtom,
        *,
        family: str,
        width: int,
    ) -> torch.Tensor:
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
        if atoms:
            rows = torch.stack(
                [
                    self._render_atom(atom, family=family, width=width)
                    for atom in atoms
                ]
            )
        else:
            rows = torch.empty((0, width), dtype=self.dtype)
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
            rows = rows[torch.randperm(rows.shape[0], generator=generator)]
        return rows.contiguous()

    def render(
        self,
        case: GraphProgramCase,
        *,
        row_permutation_seed: int = 0,
    ) -> RenderedCase:
        query = self._render_rows(
            case.query_atoms,
            family="query",
            width=self.query_dim,
            permutation_seed=row_permutation_seed,
            owner=0,
        )
        summaries = tuple(
            self._render_rows(
                node.summary_atoms,
                family="summary",
                width=self.schema.summary_dim,
                permutation_seed=row_permutation_seed,
                owner=node_id,
            )
            for node_id, node in enumerate(case.nodes)
        )
        contexts = tuple(
            self._render_rows(
                node.context_atoms,
                family="context",
                width=self.schema.context_dim,
                permutation_seed=row_permutation_seed,
                owner=node_id,
            )
            for node_id, node in enumerate(case.nodes)
        )
        edges = tuple(
            self._render_rows(
                edge.atoms,
                family="edge",
                width=self.schema.edge_dim,
                permutation_seed=row_permutation_seed,
                owner=edge_id,
            )
            for edge_id, edge in enumerate(case.edges)
        )
        return RenderedCase(
            case_id=case.case_id,
            query=query,
            summaries=summaries,
            contexts=contexts,
            edges=edges,
        )
