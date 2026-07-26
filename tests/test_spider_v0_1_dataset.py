from __future__ import annotations

from hippocampus.programs import (
    RolloutStressKind,
    build_rollout_stress_manifest,
    build_split_manifest,
    default_split_specs,
    default_split_specs_v0_2,
    generate_rollout_stress_examples,
)


def test_v0_2_split_versions_and_seeds_are_disjoint_from_v0() -> None:
    old = default_split_specs()
    follow_up = default_split_specs_v0_2()
    old_seeds = {
        seed
        for spec in old
        for seed in range(spec.seed_start, spec.seed_start + spec.case_count)
    }
    new_seeds = {
        seed
        for spec in follow_up
        for seed in range(spec.seed_start, spec.seed_start + spec.case_count)
    }

    assert not old_seeds & new_seeds
    assert {
        spec.generator_version for spec in follow_up
    } == {"spider-programs-v0.2"}
    assert sum(spec.sealed for spec in follow_up) == 1


def test_v0_2_manifests_are_deterministic_and_differ_from_v0() -> None:
    follow_up = default_split_specs_v0_2(case_scale=0.0625)
    first = tuple(build_split_manifest(spec) for spec in follow_up)
    second = tuple(build_split_manifest(spec) for spec in follow_up)
    old = build_split_manifest(default_split_specs(case_scale=0.0625)[0])

    assert first == second
    assert first[0].sha256 != old.sha256
    assert len({manifest.sha256 for manifest in first}) == len(first)


def test_rollout_stress_manifest_covers_every_registered_state() -> None:
    spec = next(
        spec
        for spec in default_split_specs_v0_2(case_scale=0.0625)
        if spec.name == "development_rollout_stress"
    )
    examples = generate_rollout_stress_examples(spec)
    first = build_rollout_stress_manifest(spec)
    second = build_rollout_stress_manifest(spec)

    assert {example.kind for example in examples} == set(RolloutStressKind)
    assert first == second
    assert len(first.descriptors) == spec.case_count
    assert all(
        example.case.case_id == example.descriptor["case_id"]
        for example in examples
    )


def test_duplicate_stress_state_preserves_frontier_occurrences() -> None:
    spec = next(
        spec
        for spec in default_split_specs_v0_2(case_scale=0.0625)
        if spec.name == "development_rollout_stress"
    )
    example = next(
        example
        for example in generate_rollout_stress_examples(spec)
        if example.kind is RolloutStressKind.DUPLICATE_CONVERGING
    )

    assert len(example.frontier_nodes) > len(set(example.frontier_nodes))
