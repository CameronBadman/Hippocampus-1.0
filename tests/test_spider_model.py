from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    GeneratorConfig,
    GraphProgramGenerator,
    ProgramFamily,
    RenderedCase,
    SyntheticManifoldRenderer,
    pack_rendered_cases,
)
from hippocampus.spider import (
    FlatTransformerScorer,
    PooledScorer,
    SpiderModel,
    SpiderModelConfig,
    padded_family_gather,
)


def _case_batch(
    *,
    family: ProgramFamily = ProgramFamily.REACHABILITY,
    seed: int = 40,
    row_seed: int = 1,
    requires_grad: bool = False,
):
    schema = GraphSchema(summary_dim=12, context_dim=12, edge_dim=12)
    generator = GraphProgramGenerator(
        GeneratorConfig(min_nodes=8, max_nodes=10)
    )
    case = generator.generate(family=family, seed=seed, answerable=True)
    renderer = SyntheticManifoldRenderer(schema, query_dim=12, seed=3)
    rendered = renderer.render(case, row_permutation_seed=row_seed)
    if requires_grad:
        rendered = RenderedCase(
            case_id=rendered.case_id,
            query=rendered.query.detach().clone().requires_grad_(),
            summaries=tuple(
                value.detach().clone().requires_grad_()
                for value in rendered.summaries
            ),
            contexts=tuple(
                value.detach().clone().requires_grad_()
                for value in rendered.contexts
            ),
            edges=tuple(
                value.detach().clone().requires_grad_()
                for value in rendered.edges
            ),
        )
    return (
        case,
        rendered,
        pack_rendered_cases((case,), (rendered,), schema=schema),
    )


def _config(*, edge_mode: str = "standard") -> SpiderModelConfig:
    return SpiderModelConfig(
        summary_dim=12,
        context_dim=12,
        edge_dim=12,
        query_dim=12,
        d_model=32,
        num_heads=4,
        num_blocks=2,
        path_rows=4,
        evidence_rows=4,
        edge_mode=edge_mode,
        edge_transforms=4,
        adapter_rank=8,
        dropout=0.0,
    )


def test_padded_family_gather_retains_empty_selection_positions() -> None:
    case, _, batch = _case_batch(family=ProgramFamily.LOOKUP)
    empty_node = next(
        node_id
        for node_id, node in enumerate(case.nodes)
        if not node.context_atoms
    )
    nonempty_node = next(
        node_id
        for node_id, node in enumerate(case.nodes)
        if node.context_atoms
    )

    padded = padded_family_gather(
        batch.graph.contexts,
        torch.tensor([empty_node, nonempty_node, empty_node]),
    )

    assert padded.values.shape[0] == 3
    assert not padded.mask[0].any()
    assert padded.mask[1].any()
    assert not padded.mask[2].any()


@pytest.mark.parametrize("edge_mode", ["standard", "compositional"])
def test_spider_candidate_shapes_and_finite_outputs(edge_mode: str) -> None:
    _, _, batch = _case_batch()
    model = SpiderModel(_config(edge_mode=edge_mode)).eval()
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    expansion = batch.graph.expand_frontier(hypotheses.node_ids)

    outputs = model.score_candidates(
        batch,
        hypotheses,
        expansion,
        evidence,
    )

    candidate_count = expansion.total_arcs
    assert outputs.next_path_state.shape == (candidate_count, 4, 32)
    assert outputs.priority_logits.shape == (candidate_count,)
    assert outputs.expand_logits.shape == (candidate_count,)
    assert outputs.context_logits.shape == (candidate_count,)
    assert outputs.evidence_logits.shape == (candidate_count,)
    assert outputs.remaining_cost.shape == (candidate_count,)
    assert outputs.support_logits.shape == (candidate_count,)
    assert outputs.conflict_logits.shape == (candidate_count,)
    for tensor in outputs.tensors():
        assert torch.isfinite(tensor).all()


def test_empty_frontier_has_canonical_model_shapes() -> None:
    _, _, batch = _case_batch()
    model = SpiderModel(_config()).eval()
    hypotheses = model.empty_hypotheses(batch.device)
    evidence = model.initial_evidence(batch)
    expansion = batch.graph.expand_frontier(hypotheses.node_ids)

    outputs = model.score_candidates(
        batch,
        hypotheses,
        expansion,
        evidence,
    )

    assert outputs.next_path_state.shape == (0, 4, 32)
    assert all(tensor.numel() == 0 for tensor in outputs.tensors()[1:])


def test_recurrent_processor_is_weight_shared_and_has_no_positions() -> None:
    model = SpiderModel(_config())

    assert model.processor_for_round(0) is model.processor_for_round(7)
    parameter_names = tuple(name.lower() for name, _ in model.named_parameters())
    assert not any("position" in name or "positional" in name for name in parameter_names)


def test_row_permutation_leaves_candidate_decisions_unchanged() -> None:
    torch.manual_seed(11)
    case, _, first_batch = _case_batch(row_seed=31)
    schema = first_batch.graph.schema
    renderer = SyntheticManifoldRenderer(schema, query_dim=12, seed=3)
    second_rendered = renderer.render(case, row_permutation_seed=91)
    second_batch = pack_rendered_cases(
        (case,),
        (second_rendered,),
        schema=schema,
    )
    model = SpiderModel(_config()).eval()

    def score(batch):
        hypotheses = model.initial_hypotheses(batch)
        expansion = batch.graph.expand_frontier(hypotheses.node_ids)
        outputs = model.score_candidates(
            batch,
            hypotheses,
            expansion,
            model.initial_evidence(batch),
        )
        return expansion.arc_ids, outputs

    first_arcs, first = score(first_batch)
    second_arcs, second = score(second_batch)
    assert torch.equal(first_arcs, second_arcs)
    assert torch.allclose(first.priority_logits, second.priority_logits, atol=2e-5)
    assert torch.allclose(first.expand_logits, second.expand_logits, atol=2e-5)
    assert torch.allclose(first.next_path_state, second.next_path_state, atol=2e-5)


def test_gradients_flow_through_packed_candidate_gathers() -> None:
    _, rendered, batch = _case_batch(requires_grad=True)
    model = SpiderModel(_config())
    hypotheses = model.initial_hypotheses(batch)
    duplicate_nodes = hypotheses.node_ids.repeat(2)
    duplicate_hypotheses = hypotheses.repeat_occurrences(
        torch.tensor([0, 0], dtype=torch.int64)
    )
    expansion = batch.graph.expand_frontier(duplicate_nodes)
    outputs = model.score_candidates(
        batch,
        duplicate_hypotheses,
        expansion,
        model.initial_evidence(batch),
    )

    loss = sum(tensor.float().sum() for tensor in outputs.tensors())
    loss.backward()

    assert rendered.query.grad is not None
    assert rendered.query.grad.abs().sum() > 0
    assert any(
        value.grad is not None and value.grad.abs().sum() > 0
        for value in rendered.summaries
    )
    assert any(
        value.grad is not None and value.grad.abs().sum() > 0
        for value in rendered.edges
    )


def test_all_empty_context_refinement_is_safe() -> None:
    case, _, batch = _case_batch(family=ProgramFamily.LOOKUP)
    model = SpiderModel(_config()).eval()
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    expansion = batch.graph.expand_frontier(hypotheses.node_ids)
    outputs = model.score_candidates(batch, hypotheses, expansion, evidence)
    empty_candidates = torch.tensor(
        [
            index
            for index, node_id in enumerate(expansion.destination_node_ids.tolist())
            if not case.nodes[node_id].context_atoms
        ],
        dtype=torch.int64,
    )
    if empty_candidates.numel() == 0:
        pytest.skip("generated frontier has no empty destination context")

    refined = model.refine_with_context(
        batch,
        expansion,
        outputs,
        empty_candidates,
    )

    assert torch.allclose(refined.next_path_state, outputs.next_path_state)
    assert torch.allclose(refined.priority_logits, outputs.priority_logits)


@pytest.mark.parametrize("scorer_type", [PooledScorer, FlatTransformerScorer])
def test_baseline_scorers_implement_common_interface(scorer_type) -> None:
    _, _, batch = _case_batch()
    model = scorer_type(_config()).eval()
    hypotheses = model.initial_hypotheses(batch)
    expansion = batch.graph.expand_frontier(hypotheses.node_ids)
    outputs = model.score_candidates(
        batch,
        hypotheses,
        expansion,
        model.initial_evidence(batch),
    )

    assert outputs.candidate_count == expansion.total_arcs
    assert outputs.next_path_state.shape[-2:] == (4, 32)


def test_checkpoint_state_round_trip_is_exact(tmp_path) -> None:
    _, _, batch = _case_batch()
    torch.manual_seed(5)
    source = SpiderModel(_config()).eval()
    hypotheses = source.initial_hypotheses(batch)
    expansion = batch.graph.expand_frontier(hypotheses.node_ids)
    expected = source.score_candidates(
        batch,
        hypotheses,
        expansion,
        source.initial_evidence(batch),
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(source.state_dict(), checkpoint)

    restored = SpiderModel(_config()).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    actual = restored.score_candidates(
        batch,
        restored.initial_hypotheses(batch),
        expansion,
        restored.initial_evidence(batch),
    )

    assert torch.equal(expected.priority_logits, actual.priority_logits)
    assert torch.equal(expected.next_path_state, actual.next_path_state)
