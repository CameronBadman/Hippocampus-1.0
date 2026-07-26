from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .renderer import RenderedCase
from .schema import GraphProgramCase


@dataclass(frozen=True, slots=True)
class FunctionalEdgeSwap:
    rendered: RenderedCase
    latent_edge_ids: tuple[int, ...]
    receiving_edge_ids: tuple[int, ...]
    donor_edge_ids: tuple[int, ...]


def swap_aligned_edge_manifolds(
    receiving_case: GraphProgramCase,
    receiving: RenderedCase,
    donor_case: GraphProgramCase,
    donor: RenderedCase,
    *,
    latent_edge_ids: Iterable[int] | None = None,
) -> FunctionalEdgeSwap:
    """Swap behaviourally aligned edge rows without comparing coordinates."""

    if receiving_case.base_case_id != donor_case.base_case_id:
        raise ValueError("functional swapping requires aligned equivalent views")
    if receiving.case_id != receiving_case.case_id:
        raise ValueError("receiving rendering does not match its case")
    if donor.case_id != donor_case.case_id:
        raise ValueError("donor rendering does not match its case")
    receiving_by_latent = {
        edge.latent_id: edge_id
        for edge_id, edge in enumerate(receiving_case.edges)
    }
    donor_by_latent = {
        edge.latent_id: edge_id
        for edge_id, edge in enumerate(donor_case.edges)
    }
    selected = (
        tuple(sorted(receiving_by_latent))
        if latent_edge_ids is None
        else tuple(latent_edge_ids)
    )
    if len(set(selected)) != len(selected):
        raise ValueError("latent edge swap IDs must be unique")
    receiving_ids: list[int] = []
    donor_ids: list[int] = []
    edges = list(receiving.edges)
    for latent_id in selected:
        if latent_id not in receiving_by_latent or latent_id not in donor_by_latent:
            raise ValueError(f"latent edge {latent_id} is not aligned across views")
        receiving_id = receiving_by_latent[latent_id]
        donor_id = donor_by_latent[latent_id]
        donor_rows = donor.edges[donor_id]
        receiving_rows = receiving.edges[receiving_id]
        if donor_rows.shape[1:] != receiving_rows.shape[1:]:
            raise ValueError("aligned edge manifolds must share family width")
        if donor_rows.device != receiving_rows.device:
            raise ValueError("edge swap does not move values across devices")
        if donor_rows.dtype != receiving_rows.dtype:
            raise ValueError("edge swap does not cast values")
        edges[receiving_id] = donor_rows
        receiving_ids.append(receiving_id)
        donor_ids.append(donor_id)
    return FunctionalEdgeSwap(
        rendered=RenderedCase(
            case_id=receiving.case_id,
            query=receiving.query,
            summaries=receiving.summaries,
            contexts=receiving.contexts,
            edges=tuple(edges),
        ),
        latent_edge_ids=selected,
        receiving_edge_ids=tuple(receiving_ids),
        donor_edge_ids=tuple(donor_ids),
    )
