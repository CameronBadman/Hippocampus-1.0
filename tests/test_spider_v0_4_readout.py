from __future__ import annotations

import pytest
import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    GraphProgramGenerator,
    ProgramFamily,
    SyntheticManifoldRenderer,
    pack_rendered_cases,
)
from hippocampus.spider import PooledScorer, SpiderModel, SpiderModelConfig


def _batch(*, row_seed: int = 0):
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    case = GraphProgramGenerator().generate(
        family=ProgramFamily.LOOKUP,
        seed=811,
        answerable=True,
    )
    renderer = SyntheticManifoldRenderer(
        schema,
        query_dim=8,
        seed=91,
        geometry="orthogonal_aligned",
    )
    return pack_rendered_cases(
        (case,),
        (renderer.render(case, row_permutation_seed=row_seed),),
        schema=schema,
    )


def _config(mode: str) -> SpiderModelConfig:
    return SpiderModelConfig(
        summary_dim=8,
        context_dim=8,
        edge_dim=8,
        query_dim=8,
        d_model=16,
        num_heads=4,
        num_blocks=1,
        path_rows=3,
        evidence_rows=3,
        dropout=0.0,
        evidence_readout=mode,
    )


@pytest.mark.parametrize(
    "model_type,mode",
    [
        (PooledScorer, "shared"),
        (PooledScorer, "dedicated_pooled"),
        (SpiderModel, "shared"),
        (SpiderModel, "dedicated_pooled"),
        (SpiderModel, "slot_aware"),
    ],
)
def test_registered_readouts_return_aligned_candidate_shapes(
    model_type,
    mode: str,
) -> None:
    batch = _batch()
    model = model_type(_config(mode)).eval()
    hypotheses = model.initial_hypotheses(batch)
    expansion = batch.graph.expand_frontier(hypotheses.node_ids)

    outputs = model.score_candidates(
        batch,
        hypotheses,
        expansion,
        model.initial_evidence(batch),
    )

    assert outputs.evidence_logits.shape == (expansion.total_arcs,)
    assert outputs.next_path_state.shape == (
        expansion.total_arcs,
        3,
        16,
    )


def test_dedicated_evidence_head_receives_its_own_gradient() -> None:
    batch = _batch()
    model = PooledScorer(_config("dedicated_pooled"))
    hypotheses = model.initial_hypotheses(batch)
    expansion = batch.graph.expand_frontier(hypotheses.node_ids)
    outputs = model.score_candidates(
        batch,
        hypotheses,
        expansion,
        model.initial_evidence(batch),
    )

    outputs.evidence_logits.sum().backward()

    assert model.evidence_readout is not None
    assert all(
        parameter.grad is not None
        for parameter in model.evidence_readout.parameters()
    )
    assert model.policy_heads.network[-1].weight.grad is None


def test_slot_aware_head_uses_rows_without_position_parameters() -> None:
    batch = _batch()
    model = SpiderModel(_config("slot_aware"))
    hypotheses = model.initial_hypotheses(batch)
    expansion = batch.graph.expand_frontier(hypotheses.node_ids)
    outputs = model.score_candidates(
        batch,
        hypotheses,
        expansion,
        model.initial_evidence(batch),
    )
    outputs.next_path_state.retain_grad()

    outputs.evidence_logits.sum().backward()

    gradient = outputs.next_path_state.grad
    assert gradient is not None
    assert not torch.allclose(gradient[:, 0], gradient[:, 1])
    assert not any(
        "position" in name.lower() or "row_index" in name.lower()
        for name, _ in model.named_parameters()
    )


def test_slot_aware_evidence_logits_are_row_permutation_invariant() -> None:
    torch.manual_seed(17)
    base = _batch(row_seed=1)
    permuted = _batch(row_seed=999)
    model = SpiderModel(_config("slot_aware")).eval()

    def score(batch):
        hypotheses = model.initial_hypotheses(batch)
        expansion = batch.graph.expand_frontier(hypotheses.node_ids)
        return model.score_candidates(
            batch,
            hypotheses,
            expansion,
            model.initial_evidence(batch),
        ).evidence_logits

    assert torch.allclose(score(base), score(permuted), atol=1e-6, rtol=1e-6)


def test_invalid_evidence_readout_is_rejected() -> None:
    with pytest.raises(ValueError, match="evidence_readout"):
        _config("mean_position_slots")
