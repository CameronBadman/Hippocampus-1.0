from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn

from ..programs.batching import PackedProgramBatch
from ..segmented import segment_mean
from ..topology import FrontierExpansion
from .arc_processor import ArcProcessor
from .config import SpiderModelConfig
from .context_refiner import ContextRefiner
from .evidence import EvidenceUpdater
from .hypothesis import HypothesisBatch
from .multiset import CrossSetRead
from .policy_heads import CandidatePolicyHeads
from .set_attention import (
    masked_mean,
    padded_family_gather,
    project_padded_set,
)
from .terminator import (
    HierarchicalTerminationHead,
    TerminationHead,
    TerminationOutput,
)
from .types import CandidateOutputs, PaddedSet


class CandidateScorerBase(nn.Module, ABC):
    """Shared packed-gather, state, context, evidence, and termination plumbing."""

    def __init__(self, config: SpiderModelConfig) -> None:
        super().__init__()
        self.config = config
        self.query_projection = nn.Linear(config.query_dim, config.d_model)
        self.summary_projection = nn.Linear(config.summary_dim, config.d_model)
        self.context_projection = nn.Linear(config.context_dim, config.d_model)
        self.edge_projection = nn.Linear(config.edge_dim, config.d_model)
        self.path_seed = nn.Parameter(
            torch.empty(config.path_rows, config.d_model)
        )
        self.evidence_seed = nn.Parameter(
            torch.empty(config.evidence_rows, config.d_model)
        )
        nn.init.normal_(self.path_seed, std=0.02)
        nn.init.normal_(self.evidence_seed, std=0.02)
        self.path_initializer = CrossSetRead(
            config.d_model,
            config.num_heads,
            config.dropout,
        )
        self.evidence_initializer = CrossSetRead(
            config.d_model,
            config.num_heads,
            config.dropout,
        )
        self.policy_heads = CandidatePolicyHeads(config.d_model)
        self.context_refiner = ContextRefiner(
            config.d_model,
            config.num_heads,
            config.dropout,
        )
        self.evidence_updater = EvidenceUpdater(
            config.d_model,
            config.num_heads,
            config.dropout,
        )
        self.termination_head = (
            TerminationHead(config.d_model, config.control_width)
            if config.termination_mode == "flat"
            else HierarchicalTerminationHead(
                config.d_model,
                config.control_width,
            )
        )

    def _validate_batch_widths(self, batch: PackedProgramBatch) -> None:
        actual = (
            batch.graph.summaries.width,
            batch.graph.contexts.width,
            batch.graph.edges.width,
            batch.query.width,
        )
        expected = (
            self.config.summary_dim,
            self.config.context_dim,
            self.config.edge_dim,
            self.config.query_dim,
        )
        if actual != expected:
            raise ValueError(
                f"packed family widths {actual} disagree with model widths {expected}"
            )

    def _queries(self, batch: PackedProgramBatch, graph_ids: torch.Tensor) -> PaddedSet:
        return project_padded_set(
            padded_family_gather(batch.query, graph_ids, validate_ids=False),
            self.query_projection,
        )

    def _candidate_sets(
        self,
        batch: PackedProgramBatch,
        expansion: FrontierExpansion,
        evidence: torch.Tensor,
    ) -> tuple[PaddedSet, PaddedSet, PaddedSet, PaddedSet, PaddedSet, torch.Tensor]:
        graph_ids = batch.graph.topology.node_graph_ids[
            expansion.source_node_ids.to(torch.int64)
        ]
        query = self._queries(batch, graph_ids)
        source = project_padded_set(
            padded_family_gather(
                batch.graph.summaries,
                expansion.source_node_ids,
                validate_ids=False,
            ),
            self.summary_projection,
        )
        edge = project_padded_set(
            padded_family_gather(
                batch.graph.edges,
                expansion.edge_ids,
                validate_ids=False,
            ),
            self.edge_projection,
        )
        destination = project_padded_set(
            padded_family_gather(
                batch.graph.summaries,
                expansion.destination_node_ids,
                validate_ids=False,
            ),
            self.summary_projection,
        )
        evidence_values = evidence[graph_ids.to(torch.int64)]
        evidence_set = PaddedSet(
            evidence_values,
            torch.ones(
                evidence_values.shape[:2],
                dtype=torch.bool,
                device=evidence_values.device,
            ),
        )
        return query, source, edge, destination, evidence_set, graph_ids

    def initial_hypotheses(self, batch: PackedProgramBatch) -> HypothesisBatch:
        self._validate_batch_widths(batch)
        graph_ids: list[int] = []
        node_ids: list[int] = []
        for graph_id, case in enumerate(batch.cases):
            node_offset = int(
                batch.graph.topology.graph_node_ptr[graph_id].item()
            )
            for local_node in case.start_nodes:
                graph_ids.append(graph_id)
                node_ids.append(node_offset + local_node)
        device = batch.device
        graph_tensor = torch.tensor(graph_ids, dtype=torch.int32, device=device)
        node_tensor = torch.tensor(node_ids, dtype=torch.int32, device=device)
        query = self._queries(batch, graph_tensor)
        path = self.path_seed.to(device=device).unsqueeze(0).expand(
            len(node_ids),
            -1,
            -1,
        )
        path_mask = torch.ones(
            path.shape[:2],
            dtype=torch.bool,
            device=device,
        )
        path = self.path_initializer(path, path_mask, query)
        hypotheses = HypothesisBatch(
            node_ids=node_tensor,
            graph_ids=graph_tensor,
            path_state=path,
            scores=path.new_zeros((len(node_ids),)),
            depths=torch.zeros(len(node_ids), dtype=torch.int32, device=device),
            parent_trace_ids=torch.full(
                (len(node_ids),),
                -1,
                dtype=torch.int64,
                device=device,
            ),
            incoming_arc_ids=torch.full(
                (len(node_ids),),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            incoming_edge_ids=torch.full(
                (len(node_ids),),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            context_read=torch.zeros(
                len(node_ids),
                dtype=torch.bool,
                device=device,
            ),
        )
        return hypotheses.validate()

    def empty_hypotheses(self, device: torch.device | str) -> HypothesisBatch:
        target = torch.device(device)
        reference = self.path_seed.to(target)
        return HypothesisBatch(
            node_ids=torch.empty(0, dtype=torch.int32, device=target),
            graph_ids=torch.empty(0, dtype=torch.int32, device=target),
            path_state=reference.new_empty(
                (0, self.config.path_rows, self.config.d_model)
            ),
            scores=reference.new_empty((0,)),
            depths=torch.empty(0, dtype=torch.int32, device=target),
            parent_trace_ids=torch.empty(0, dtype=torch.int64, device=target),
            incoming_arc_ids=torch.empty(0, dtype=torch.int32, device=target),
            incoming_edge_ids=torch.empty(0, dtype=torch.int32, device=target),
            context_read=torch.empty(0, dtype=torch.bool, device=target),
        )

    def initial_evidence(self, batch: PackedProgramBatch) -> torch.Tensor:
        graph_count = batch.graph_count
        evidence = self.evidence_seed.to(batch.device).unsqueeze(0).expand(
            graph_count,
            -1,
            -1,
        )
        if not self.config.use_global_evidence:
            return torch.zeros_like(evidence)
        graph_ids = torch.arange(
            graph_count,
            dtype=torch.int32,
            device=batch.device,
        )
        query = self._queries(batch, graph_ids)
        evidence_mask = torch.ones(
            evidence.shape[:2],
            dtype=torch.bool,
            device=batch.device,
        )
        return self.evidence_initializer(evidence, evidence_mask, query)

    @abstractmethod
    def score_candidates(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        expansion: FrontierExpansion,
        evidence: torch.Tensor,
        controller_features: torch.Tensor | None = None,
        *,
        round_index: int = 0,
    ) -> CandidateOutputs:
        raise NotImplementedError

    def refine_with_context(
        self,
        batch: PackedProgramBatch,
        expansion: FrontierExpansion,
        outputs: CandidateOutputs,
        candidate_indices: torch.Tensor,
    ) -> CandidateOutputs:
        indices = candidate_indices.to(device=batch.device, dtype=torch.int64)
        if indices.numel() == 0:
            return outputs
        destinations = expansion.destination_node_ids[indices]
        context = project_padded_set(
            padded_family_gather(
                batch.graph.contexts,
                destinations,
                validate_ids=False,
            ),
            self.context_projection,
        )
        selected_path = outputs.next_path_state[indices]
        refined_path = self.context_refiner(selected_path, context)
        refined_outputs = self.policy_heads(refined_path)
        return outputs.index_copy(indices, refined_outputs)

    def update_evidence(
        self,
        evidence: torch.Tensor,
        messages: torch.Tensor,
        graph_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.use_global_evidence or messages.shape[0] == 0:
            return evidence
        return self.evidence_updater(evidence, messages, graph_ids)

    def termination_output(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        evidence: torch.Tensor,
        controller_features: torch.Tensor | None = None,
    ) -> TerminationOutput:
        graph_count = batch.graph_count
        graph_ids = torch.arange(
            graph_count,
            dtype=torch.int32,
            device=batch.device,
        )
        query_set = self._queries(batch, graph_ids)
        query = masked_mean(query_set.values, query_set.mask)
        evidence_pool = (
            evidence.mean(dim=1)
            if self.config.use_global_evidence
            else torch.zeros(
                (graph_count, self.config.d_model),
                dtype=query.dtype,
                device=query.device,
            )
        )
        frontier_rows = hypotheses.path_state.mean(dim=1)
        frontier = segment_mean(
            frontier_rows,
            row_owner_ids=hypotheses.graph_ids,
            num_segments=graph_count,
        ).values
        control = (
            query.new_zeros((graph_count, self.config.control_width))
            if controller_features is None
            else controller_features
        )
        output = self.termination_head(
            query,
            evidence_pool,
            frontier,
            control,
        )
        return (
            TerminationOutput(logits=output)
            if isinstance(output, torch.Tensor)
            else output
        )

    def termination_logits(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        evidence: torch.Tensor,
        controller_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.termination_output(
            batch,
            hypotheses,
            evidence,
            controller_features,
        ).logits


class SpiderModel(CandidateScorerBase):
    def __init__(self, config: SpiderModelConfig) -> None:
        super().__init__(config)
        if config.tied_recurrence:
            self.arc_processors = nn.ModuleList([ArcProcessor(config)])
        else:
            self.arc_processors = nn.ModuleList(
                [ArcProcessor(config) for _ in range(config.untied_rounds)]
            )

    def processor_for_round(self, round_index: int) -> ArcProcessor:
        if self.config.tied_recurrence:
            return self.arc_processors[0]
        return self.arc_processors[min(round_index, len(self.arc_processors) - 1)]

    def score_candidates(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        expansion: FrontierExpansion,
        evidence: torch.Tensor,
        controller_features: torch.Tensor | None = None,
        *,
        round_index: int = 0,
    ) -> CandidateOutputs:
        self._validate_batch_widths(batch)
        if expansion.total_arcs == 0:
            return self.policy_heads.empty(
                reference=self.path_seed,
                path_rows=self.config.path_rows,
                d_model=self.config.d_model,
            )
        parent_positions = expansion.frontier_positions.to(torch.int64)
        path = hypotheses.path_state[parent_positions]
        query, source, edge, destination, evidence_set, _ = self._candidate_sets(
            batch,
            expansion,
            evidence,
        )
        control = (
            path.new_zeros((expansion.total_arcs, self.config.control_width))
            if controller_features is None
            else controller_features
        )
        path, _ = self.processor_for_round(round_index)(
            path,
            query,
            evidence_set,
            source,
            edge,
            destination,
            control,
        )
        return self.policy_heads(path)
