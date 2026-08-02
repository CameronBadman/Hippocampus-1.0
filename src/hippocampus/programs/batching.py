from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch

from ..config import GraphSchema, PackConfig
from ..graph import PackedGraph, pack_graph_from_topology
from ..manifold import PackedManifoldFamily, pack_manifold_family
from ..topology import TopologyComponent, pack_topology
from .renderer import RenderedCase
from .renderer import SyntheticManifoldRenderer
from .schema import GraphProgramCase


@dataclass(frozen=True, slots=True)
class PackedProgramBatch:
    graph: PackedGraph
    query: PackedManifoldFamily
    cases: tuple[GraphProgramCase, ...]

    @property
    def device(self) -> torch.device:
        return self.graph.device

    @property
    def graph_count(self) -> int:
        return len(self.cases)

    def global_node_id(self, graph_id: int, local_node_id: int) -> int:
        start = int(self.graph.topology.graph_node_ptr[graph_id].item())
        return start + local_node_id

    def global_edge_id(self, graph_id: int, local_edge_id: int) -> int:
        start = int(self.graph.topology.graph_edge_ptr[graph_id].item())
        return start + local_edge_id


class FreshRenderedBatchSource(Sequence[PackedProgramBatch]):
    """Lazily rerender one case with a fresh deterministic row order.

    Presentation counters are explicit checkpoint state. This keeps exact
    training resume possible while avoiding reuse of one pre-rendered row
    ordering. The source is intentionally single-process; callers that add
    concurrent data workers must coordinate counter ownership themselves.
    """

    state_format = "fresh-rendered-batch-source-v1"

    def __init__(
        self,
        cases: Sequence[GraphProgramCase],
        *,
        renderer: SyntheticManifoldRenderer,
        schema: GraphSchema,
        base_row_seed: int,
        pack_config: PackConfig | None = None,
        validate: bool = True,
    ) -> None:
        if not cases:
            raise ValueError("a fresh batch source requires at least one case")
        self._cases = tuple(cases)
        self.renderer = renderer
        self.schema = schema
        self.base_row_seed = int(base_row_seed)
        self.pack_config = pack_config
        self.validate = validate
        self._presentation_counts = [0] * len(self._cases)
        payload = "|".join(
            (
                self.state_format,
                *(case.case_id for case in self._cases),
                renderer.renderer_version,
                renderer.geometry,
                str(renderer.seed),
                str(self.base_row_seed),
                repr(schema),
                repr(pack_config),
            )
        )
        self._source_id = hashlib.sha256(payload.encode()).hexdigest()

    def __len__(self) -> int:
        return len(self._cases)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self._cases)

    @property
    def presentation_counts(self) -> tuple[int, ...]:
        return tuple(self._presentation_counts)

    def _row_seed(self, index: int, presentation: int) -> int:
        digest = hashlib.sha256(
            (
                f"{self.base_row_seed}|{self._cases[index].case_id}|"
                f"{index}|{presentation}"
            ).encode()
        ).digest()
        return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)

    def __getitem__(self, index: int) -> PackedProgramBatch:
        if not isinstance(index, int):
            raise TypeError("fresh batch source indices must be integers")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        presentation = self._presentation_counts[index]
        self._presentation_counts[index] += 1
        case = self._cases[index]
        rendered = self.renderer.render(
            case,
            row_permutation_seed=self._row_seed(index, presentation),
        )
        return pack_rendered_cases(
            (case,),
            (rendered,),
            schema=self.schema,
            pack_config=self.pack_config,
            validate=self.validate,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "format": self.state_format,
            "source_id": self._source_id,
            "presentation_counts": tuple(self._presentation_counts),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("format") != self.state_format:
            raise ValueError("fresh batch source checkpoint format mismatch")
        if state.get("source_id") != self._source_id:
            raise ValueError("fresh batch source checkpoint identity mismatch")
        counts = tuple(int(value) for value in state["presentation_counts"])
        if len(counts) != len(self) or any(value < 0 for value in counts):
            raise ValueError("fresh batch source presentation state is invalid")
        self._presentation_counts[:] = counts


def pack_rendered_cases(
    cases: Sequence[GraphProgramCase],
    rendered: Sequence[RenderedCase],
    *,
    schema: GraphSchema,
    pack_config: PackConfig | None = None,
    validate: bool = True,
) -> PackedProgramBatch:
    if len(cases) != len(rendered):
        raise ValueError("cases and rendered values must have equal lengths")
    if not cases:
        raise ValueError("a packed program batch requires at least one case")
    for index, (case, values) in enumerate(zip(cases, rendered, strict=True)):
        if case.case_id != values.case_id:
            raise ValueError(f"rendered case {index} does not match supervisor case")
        if len(values.summaries) != len(case.nodes):
            raise ValueError(f"rendered case {index} summaries do not align")
        if len(values.contexts) != len(case.nodes):
            raise ValueError(f"rendered case {index} contexts do not align")
        if len(values.edges) != len(case.edges):
            raise ValueError(f"rendered case {index} edges do not align")

    topology = pack_topology(
        [
            TopologyComponent(
                node_count=len(case.nodes),
                edge_src=[edge.source_node for edge in case.edges],
                edge_dst=[edge.destination_node for edge in case.edges],
                edge_bidirectional=[edge.bidirectional for edge in case.edges],
                schema=schema,
            )
            for case in cases
        ],
        device=(
            pack_config.device
            if pack_config is not None and pack_config.device is not None
            else rendered[0].query.device
        ),
        schema=schema,
        validate=validate,
    )
    summaries = tuple(
        rows for values in rendered for rows in values.summaries
    )
    contexts = tuple(
        rows for values in rendered for rows in values.contexts
    )
    edges = tuple(rows for values in rendered for rows in values.edges)
    graph = pack_graph_from_topology(
        topology,
        summaries,
        contexts,
        edges,
        pack_config=pack_config,
        schema=schema,
        validate=validate,
    )

    query_dim = rendered[0].query_dim
    if any(values.query_dim != query_dim for values in rendered):
        raise ValueError("all rendered query manifolds must share one width")
    query = pack_manifold_family(
        tuple(values.query for values in rendered),
        owner_count=len(cases),
        width=query_dim,
        owner_graph_ids=torch.arange(
            len(cases),
            dtype=torch.int32,
            device=graph.device,
        ),
        resolved_pack_config=graph.resolved_pack_config,
        allow_empty=False,
        family_name="query",
        validate=validate,
    )
    return PackedProgramBatch(
        graph=graph,
        query=query,
        cases=tuple(cases),
    )
