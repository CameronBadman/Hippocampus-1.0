from __future__ import annotations

import ast
from pathlib import Path


def _constants() -> dict[str, object]:
    source = Path(
        "scripts/colab_spider_v0_2_recurrence_worker.py"
    ).read_text()
    tree = ast.parse(source)
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


def test_colab_worker_is_pinned_and_has_no_sealed_dataset() -> None:
    constants = _constants()

    assert len(str(constants["SOURCE_COMMIT"])) == 40
    assert constants["STEPS"] == 6000
    assert constants["ALLOWED_ACCELERATORS"] == ("H100", "A100")
    assert "sealed" not in str(constants["CONFIGS"]).lower()
    assert len(str(constants["TRAIN_MANIFEST_SHA256"])) == 64
    assert len(str(constants["VALIDATION_MANIFEST_SHA256"])) == 64
