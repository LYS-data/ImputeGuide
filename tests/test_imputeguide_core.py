"""Tests for the public ImputeGuide core."""

from __future__ import annotations

import numpy as np
import pytest

from imputeguide.candidates import merge_candidate_rankings
from imputeguide.confirmation import confirm_challenger
from imputeguide.execution import execute_whole_table
from imputeguide.history import HistoricalRun, build_stable_strategy
from imputeguide.opportunity import (
    ScenarioMatch,
    block_balanced_similarity,
    rank_historical_opportunities,
    rank_probe_expansion,
)
from imputeguide.selector import select_from_structural_evidence
from imputeguide.sampling import stratified_validation_rows
from imputeguide.registry import METHODS, build_imputer
from imputeguide.structural import (
    ClusteringSeeds,
    ComponentMap,
    StructuralPerturbation,
    build_perturbation_plan,
    score_perturbation,
)


def _history(extra_copies: bool = False) -> list[HistoricalRun]:
    rows: list[HistoricalRun] = []
    utilities = {
        "alpha": {"a": (0.8, 0.6), "b": (0.7, 0.7)},
        "beta": {"a": (0.5, 0.5), "b": (0.6, 0.4)},
    }
    for dataset, methods in utilities.items():
        for method, values in methods.items():
            for index, value in enumerate(values):
                rows.append(HistoricalRun(
                    dataset, f"s{index}", method, value, 2.0, "success"
                ))
                if extra_copies and dataset == "alpha":
                    rows.append(HistoricalRun(
                        dataset, f"copy{index}", method, value, 2.0, "success"
                    ))
    return rows


def _stable(rows: list[HistoricalRun]):
    return build_stable_strategy(
        rows,
        eligible_methods=("a", "b"),
        stability_penalty=0.1,
        minimum_coverage=1.0,
        minimum_successful_scenarios=2,
        minimum_common_datasets=2,
    )


def test_history_uses_dataset_macro_aggregation() -> None:
    baseline = _stable(_history())
    duplicated = _stable(_history(extra_copies=True))
    assert baseline.method == duplicated.method == "a"
    assert baseline.macro_scores == pytest.approx(duplicated.macro_scores)


def test_failed_history_run_reduces_coverage_without_synthetic_utility() -> None:
    rows = _history()
    target = next(
        index for index, row in enumerate(rows)
        if row.dataset == "alpha" and row.method == "b" and row.scenario == "s1"
    )
    rows[target] = HistoricalRun("alpha", "s1", "b", None, 2.0, "failure")
    with pytest.raises(ValueError, match="insufficient history"):
        _stable(rows)


def test_candidate_merge_is_deduplicated_and_budget_bounded() -> None:
    candidates = merge_candidate_rankings(
        ("anchor", "a", "b", "c"),
        ("a", "d", "e"),
        anchor="anchor",
        historical_quota=2,
        probe_quota=2,
    )
    assert candidates.historical == ("a", "b")
    assert candidates.probe == ("d", "e")
    assert candidates.merged == ("a", "d", "b", "e")


def test_confirmation_is_deterministic_and_equality_retains_anchor() -> None:
    kwargs = dict(
        anchor="a",
        challenger="b",
        anchor_scores=np.full(5, 0.5),
        challenger_scores=np.full(5, 0.505),
        alpha=0.10,
        bootstrap_repeats=200,
        margin=0.005,
        seed=7,
    )
    first = confirm_challenger(**kwargs)
    second = confirm_challenger(**kwargs)
    assert first == second
    assert first.selected_method == "a"
    assert not first.switched


def test_opportunity_gate_prevents_candidate_attempts() -> None:
    result = select_from_structural_evidence(
        anchor="a",
        candidates=("b", "c", "d", "e", "f"),
        completion_success={"a": True},
        discovery_scores={},
        confirmation_scores={},
        opportunity_score=0.0,
        opportunity_threshold=0.0,
        confirmation_alpha=0.10,
        bootstrap_repeats=10,
        switch_margin=0.005,
        bootstrap_seed=1,
        full_table_budget=1,
    )
    assert result.selected_method == "a"
    assert result.attempted_methods == ("a",)


def test_failed_candidate_cannot_invalidate_anchor() -> None:
    result = select_from_structural_evidence(
        anchor="a",
        candidates=("b",),
        completion_success={"a": True, "b": False},
        discovery_scores={"a": np.array([0.4, 0.4, 0.4])},
        confirmation_scores={},
        opportunity_score=1.0,
        opportunity_threshold=0.0,
        confirmation_alpha=0.10,
        bootstrap_repeats=10,
        switch_margin=0.005,
        bootstrap_seed=1,
        full_table_budget=2,
    )
    assert result.selected_method == "a"
    assert result.stop_reason == "no_positive_discovery_gain"


def test_perturbation_namespaces_are_deterministic_and_disjoint() -> None:
    kwargs = dict(
        n_rows=20,
        n_features=4,
        discovery_perturbations=3,
        confirmation_perturbations=5,
        row_resamples=3,
        feature_subsets=2,
        row_fraction=0.8,
        feature_fraction=0.5,
        minimum_row_overlap=4,
        minimum_features=2,
        discovery_seed=11,
        confirmation_seed=29,
        feature_groups=((0, 1), (2,), (3,)),
    )
    first = build_perturbation_plan(**kwargs)
    assert first == build_perturbation_plan(**kwargs)
    identifiers = {item.identifier for item in first.discovery}
    assert identifiers.isdisjoint(item.identifier for item in first.confirmation)
    for perturbation in (*first.discovery, *first.confirmation):
        for subset in perturbation.feature_subsets:
            assert (0 in subset) == (1 in subset)


def test_structural_score_uses_intersections_and_fixed_maps() -> None:
    matrix = np.array([
        [-3.0, -2.9], [-2.0, -2.1], [-1.0, -1.1],
        [1.0, 1.1], [2.0, 2.1], [3.0, 2.9],
    ])

    def clusterer(values: np.ndarray, seed: int) -> np.ndarray:
        del seed
        return (values[:, 0] > 0).astype(int)

    perturbation = StructuralPerturbation(
        identifier="test",
        row_samples=((0, 1, 3, 4), (1, 2, 4, 5)),
        feature_subsets=((0,), (1,)),
        seeds=ClusteringSeeds(1, (2, 3), (4, 5)),
    )
    result = score_perturbation(
        matrix,
        perturbation,
        clusterer=clusterer,
        component_maps=(
            ComponentMap(-1.0, 1.0),
            ComponentMap(-1.0, 1.0),
            ComponentMap(-1.0, 1.0),
        ),
        weights=(0.40, 0.35, 0.25),
        minimum_row_overlap=2,
    )
    assert result.valid
    assert result.score is not None and 0.0 <= result.score <= 1.0
    assert result.components is not None
    assert result.components.row_consistency == pytest.approx(1.0)
    assert result.components.feature_agreement == pytest.approx(1.0)


def test_history_opportunity_aggregates_distinct_datasets() -> None:
    matches = [
        ScenarioMatch("d1", "s1", 1.0),
        ScenarioMatch("d1", "s2", 0.9),
        ScenarioMatch("d2", "s1", 0.8),
        ScenarioMatch("d2", "s2", 0.7),
    ]
    gains = {
        ("d1", "s1", "m"): 0.2,
        ("d1", "s2", "m"): 0.2,
        ("d2", "s1", "m"): 0.1,
        ("d2", "s2", "m"): 0.1,
    }
    result = rank_historical_opportunities(
        matches,
        gains,
        methods=("m",),
        variation_penalty=0.0,
        maximum_scenarios_per_dataset=2,
        topmean_scenarios=1,
        minimum_paired_coverage=1.0,
        minimum_paired_datasets=2,
        minimum_effective_sample_size=1.5,
        minimum_maximum_similarity=0.5,
        registry_order=("m",),
    )
    assert result.ranking == ("m",)
    assert result.supported[0].paired_datasets == 2
    assert 1.5 <= result.supported[0].effective_sample_size <= 2.0


def test_probe_expansion_respects_family_cap() -> None:
    result = rank_probe_expansion(
        {
            "mean": np.array([0.4, 0.4, 0.4]),
            "gain": np.array([0.6, 0.6, 0.6]),
            "knni": np.array([0.5, 0.5, 0.5]),
        },
        reference="mean",
        representative_family={
            "mean": "statistical",
            "gain": "learned",
            "knni": "neighbor",
        },
        within_family_ranking={
            "statistical": ("median",),
            "learned": ("hivae", "gain"),
            "neighbor": ("knni",),
        },
        executable_methods=("median", "hivae", "gain", "knni"),
        anchor="median",
        variation_penalty=0.5,
        total_quota=2,
        per_family_cap=1,
        registry_order=("median", "knni", "gain", "hivae"),
    )
    assert result.ranking == ("hivae", "knni")


def test_block_similarity_is_truncated_at_zero() -> None:
    assert block_balanced_similarity(
        (np.array([1.0, 0.0]),),
        (np.array([-1.0, 0.0]),),
        block_weights=(1.0,),
        epsilon=1e-12,
    ) == 0.0


def test_validation_rows_are_deterministic_and_stratified() -> None:
    mask = np.ones((20, 4), dtype=bool)
    mask[5:10, :1] = False
    mask[10:15, :2] = False
    mask[15:, :3] = False
    kwargs = dict(
        cap=8,
        training_quantile_cutpoints=np.array([0.20, 0.40, 0.60]),
        seed=42,
    )
    first = stratified_validation_rows(mask, **kwargs)
    second = stratified_validation_rows(mask, **kwargs)
    assert np.array_equal(first, second)
    assert len(first) == 8
    fractions = 1.0 - mask[first].mean(axis=1)
    assert set(fractions) == {0.0, 0.25, 0.5, 0.75}


def test_whole_table_contract_rejects_observed_cell_changes() -> None:
    incomplete = np.array([[1.0, np.nan], [2.0, 3.0]])

    class BadAdapter:
        def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
            return np.nan_to_num(matrix, nan=0.0) + 1.0

    attempt = execute_whole_table("bad", incomplete, builder=BadAdapter)
    assert attempt.status == "failure"
    assert attempt.failure_message == "completion changed observed cells"


def test_whole_table_contract_records_candidate_failure() -> None:
    incomplete = np.array([[1.0, np.nan], [2.0, 3.0]])

    class FailingAdapter:
        def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
            del matrix
            raise RuntimeError("backend unavailable")

    attempt = execute_whole_table("failed", incomplete, builder=FailingAdapter)
    assert attempt.status == "failure"
    assert attempt.completion is None
    assert attempt.failure_type == "RuntimeError"


def test_public_registry_exposes_19_supported_methods() -> None:
    assert len(METHODS) == 19
    assert len(set(METHODS)) == 19
    assert "knn_uniform" not in METHODS
    with pytest.raises(KeyError, match="unsupported imputation method"):
        build_imputer("knn_uniform")
