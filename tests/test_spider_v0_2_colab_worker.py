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


def test_launch_specs_cover_the_frozen_six_run_matrix() -> None:
    specs = {}
    worker_hashes = set()
    worker_urls = set()
    for path in sorted(
        Path("scripts/colab_spider_v0_2_specs").glob("*.py")
    ):
        tree = ast.parse(path.read_text())
        assignments = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        assignments[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        spec = assignments["RUN_SPEC"]
        specs[spec["experiment_id"]] = (spec["model"], spec["seed"])
        worker_hashes.add(assignments["EXPECTED_SHA256"])
        worker_urls.add(assignments["WORKER_URL"])

    assert specs == {
        "REC-recurrent-s1701-6k": ("recurrent", 1701),
        "REC-recurrent-s1802-6k": ("recurrent", 1802),
        "REC-recurrent-s1903-6k": ("recurrent", 1903),
        "REC-pooled-s1701-6k": ("pooled", 1701),
        "REC-pooled-s1802-6k": ("pooled", 1802),
        "REC-pooled-s1903-6k": ("pooled", 1903),
    }
    assert len(worker_hashes) == 1
    assert len(next(iter(worker_hashes))) == 64
    assert len(worker_urls) == 1
    assert "44dc478bb582224268e4f94a19b99f4681579b86" in next(
        iter(worker_urls)
    )
