from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import replace

import torch
from torch import nn

from ..programs.batching import PackedProgramBatch
from ..segmented import segment_mean
from ..topology import FrontierExpansion
from .arc_processor import ArcProcessor
from .config import SpiderModelConfig
from .context_refiner import ContextRefiner
from .evidence import EvidenceUpdater
from .evidence_readout import (
    DedicatedPooledEvidenceReadout,
    SlotAwareEvidenceReadout,
)
from .hypothesis import HypothesisBatch
from .multiset import CrossSetRead
from .policy_heads import CandidatePolicyHeads
from .set_attention import (
    masked_mean,
    padded_family_gather,
    project_padded_set,
)
from .terminator import (
    FactorizedTerminationHead,
    HierarchicalTerminationHead,
    TerminationHead,
    TerminationOutput,
)
from .types import CandidateOutputs, CandidateReadoutContext, PaddedSet


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
        if config.evidence_readout == "shared":
            self.evidence_readout = None
        elif config.evidence_readout == "dedicated_pooled":
            self.evidence_readout = DedicatedPooledEvidenceReadout(
                config.d_model
            )
        else:
            self.evidence_readout = SlotAwareEvidenceReadout(
                config.d_model,
                config.num_heads,
                config.control_width,
                config.dropout,
            )
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
        if config.termination_mode == "flat":
            self.termination_head = TerminationHead(
                config.d_model,
                config.control_width,
            )
        elif config.termination_mode == "hierarchical":
            self.termination_head = HierarchicalTerminationHead(
                config.d_model,
                config.control_width,
            )
        else:
            self.termination_head = FactorizedTerminationHead(
                config.d_model,
                config.control_width,
            )
        self.null_expansion_head = (
            nn.Sequential(
                nn.LayerNorm(3 * config.d_model + config.control_width),
                nn.Linear(
                    3 * config.d_model + config.control_width,
                    config.d_model,
                ),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )
            if config.use_null_expansion
            else None
        )
        global_width = 3 * config.d_model + config.control_width
        self.evidence_null_head = (
            nn.Sequential(
                nn.LayerNorm(global_width),
                nn.Linear(global_width, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )
            if config.use_evidence_null
            else None
        )
        self.evidence_cardinality_head = (
            nn.Sequential(
                nn.LayerNorm(global_width),
                nn.Linear(global_width, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 5),
            )
            if config.use_evidence_cardinality
            else None
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
        path = self.initial_path_state(batch, graph_tensor)
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

    def initial_path_state(
        self,
        batch: PackedProgramBatch,
        graph_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build the query-conditioned path seed for owner occurrences."""

        resolved_graph_ids = graph_ids.to(
            device=batch.device,
            dtype=torch.int32,
        )
        query = self._queries(batch, resolved_graph_ids)
        path = self.path_seed.to(device=batch.device).unsqueeze(0).expand(
            resolved_graph_ids.numel(),
            -1,
            -1,
        )
        path_mask = torch.ones(
            path.shape[:2],
            dtype=torch.bool,
            device=batch.device,
        )
        return self.path_initializer(path, path_mask, query)

    def pooled_current_node_path_state(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
    ) -> torch.Tensor:
        """Replace history with a symmetric current-summary representation."""

        summary = project_padded_set(
            padded_family_gather(
                batch.graph.summaries,
                hypotheses.node_ids,
                validate_ids=False,
            ),
            self.summary_projection,
        )
        pooled = masked_mean(summary.values, summary.mask)
        return pooled[:, None, :].expand(
            -1,
            self.config.path_rows,
            -1,
        )

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

    def _candidate_policy_outputs(
        self,
        path_state: torch.Tensor,
        context: CandidateReadoutContext,
    ) -> CandidateOutputs:
        outputs = self.policy_heads(path_state)
        if self.evidence_readout is not None:
            outputs = replace(
                outputs,
                evidence_logits=self.evidence_readout(path_state, context),
            )
        return replace(outputs, readout_context=context)

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
        readout_context = outputs.readout_context
        if readout_context is None:
            refined_outputs = self.policy_heads(refined_path)
        else:
            refined_outputs = self._candidate_policy_outputs(
                refined_path,
                readout_context.index_select(indices),
            )
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

    def _global_state_inputs(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        evidence: torch.Tensor,
        controller_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return query, evidence_pool, frontier, control

    def null_expansion_logits(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        evidence: torch.Tensor,
        controller_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score one explicit NULL action for every active hypothesis."""

        if self.null_expansion_head is None:
            raise RuntimeError(
                "null expansion was not enabled in SpiderModelConfig"
            )
        inputs = self.null_expansion_inputs(
            batch,
            hypotheses,
            evidence,
            controller_features,
        )
        return self.null_expansion_head(
            torch.cat(inputs, dim=-1)
        ).squeeze(-1)

    def null_expansion_inputs(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        evidence: torch.Tensor,
        controller_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return frozen, per-hypothesis inputs to the branch NULL head."""

        if hypotheses.count == 0:
            empty = evidence.new_empty((0, self.config.d_model))
            control = evidence.new_empty((0, self.config.control_width))
            return empty, empty, empty, control
        graph_ids = hypotheses.graph_ids.to(torch.int64)
        query_set = self._queries(batch, graph_ids)
        query = masked_mean(query_set.values, query_set.mask)
        evidence_pool = (
            evidence[graph_ids].mean(dim=1)
            if self.config.use_global_evidence
            else query.new_zeros((hypotheses.count, self.config.d_model))
        )
        path = hypotheses.path_state.mean(dim=1)
        control = (
            query.new_zeros(
                (hypotheses.count, self.config.control_width)
            )
            if controller_features is None
            else controller_features[graph_ids]
        )
        return query, evidence_pool, path, control

    def termination_inputs(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        evidence: torch.Tensor,
        controller_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pooled inputs at the canonical post-transition boundary."""

        return self._global_state_inputs(
            batch,
            hypotheses,
            evidence,
            controller_features,
        )

    def evidence_selection_logits(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        evidence: torch.Tensor,
        controller_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Predict a per-graph null score and total evidence cardinality."""

        if (
            self.evidence_null_head is None
            and self.evidence_cardinality_head is None
        ):
            return None, None
        inputs = self._global_state_inputs(
            batch,
            hypotheses,
            evidence,
            controller_features,
        )
        joined = torch.cat(inputs, dim=-1)
        null_logits = (
            None
            if self.evidence_null_head is None
            else self.evidence_null_head(joined).squeeze(-1)
        )
        cardinality_logits = (
            None
            if self.evidence_cardinality_head is None
            else self.evidence_cardinality_head(joined)
        )
        return null_logits, cardinality_logits

    def termination_output(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        evidence: torch.Tensor,
        controller_features: torch.Tensor | None = None,
    ) -> TerminationOutput:
        query, evidence_pool, frontier, control = self.termination_inputs(
            batch,
            hypotheses,
            evidence,
            controller_features,
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
        readout_context = CandidateReadoutContext(
            query=query,
            source=source,
            edge=edge,
            destination=destination,
            global_evidence=evidence_set,
            controller_features=control,
        )
        return self._candidate_policy_outputs(path, readout_context)
