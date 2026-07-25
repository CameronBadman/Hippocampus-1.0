from __future__ import annotations

import pytest
import torch

from hippocampus import GraphSchema, PackConfig
from hippocampus.programs import (
    GeneratorConfig,
    GraphProgramGenerator,
    ProgramFamily,
    SyntheticManifoldRenderer,
    pack_rendered_cases,
)
from hippocampus.spider import SpiderModel, SpiderModelConfig


pytestmark = pytest.mark.cuda


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not visible to this test process")


@pytest.mark.parametrize(
    ("dtype", "pack_config"),
    [
        (torch.float32, PackConfig.cuda_fp32()),
        (torch.bfloat16, PackConfig.cuda_bf16()),
    ],
)
def test_cuda_candidate_training_path(
    dtype: torch.dtype,
    pack_config: PackConfig,
) -> None:
    _require_cuda()
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("selected CUDA device does not support BF16")
    schema = GraphSchema(summary_dim=16, context_dim=16, edge_dim=16)
    generator = GraphProgramGenerator(
        GeneratorConfig(min_nodes=8, max_nodes=10)
    )
    cases = tuple(
        generator.generate(
            family=family,
            seed=700 + index,
            answerable=True,
        )
        for index, family in enumerate(ProgramFamily)
    )
    renderer = SyntheticManifoldRenderer(schema, query_dim=16, seed=33)
    rendered = tuple(renderer.render(case) for case in cases)
    batch = pack_rendered_cases(
        cases,
        rendered,
        schema=schema,
        pack_config=pack_config,
    )
    model = SpiderModel(
        SpiderModelConfig(
            summary_dim=16,
            context_dim=16,
            edge_dim=16,
            query_dim=16,
            d_model=32,
            num_heads=4,
            num_blocks=1,
            path_rows=4,
            evidence_rows=4,
            edge_mode="compositional",
            edge_transforms=4,
            adapter_rank=8,
            dropout=0.0,
        )
    ).to(device="cuda", dtype=dtype)
    hypotheses = model.initial_hypotheses(batch)
    expansion = batch.graph.expand_frontier(hypotheses.node_ids)
    outputs = model.score_candidates(
        batch,
        hypotheses,
        expansion,
        model.initial_evidence(batch),
    )
    loss = (
        outputs.priority_logits.float().square().mean()
        + outputs.expand_logits.float().square().mean()
        + outputs.next_path_state.float().square().mean()
    )
    loss.backward()

    assert batch.device.type == "cuda"
    assert batch.graph.dtype is dtype
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )
