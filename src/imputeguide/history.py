"""Stable Strategy construction from aligned historical evaluations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Iterable, Sequence


SUCCESS = "success"


@dataclass(frozen=True, slots=True)
class HistoricalRun:
    """One registered historical dataset/scenario/method execution."""

    dataset: str
    scenario: str
    method: str
    utility: float | None
    runtime_seconds: float
    status: str


@dataclass(frozen=True, slots=True)
class DatasetMethodSummary:
    dataset: str
    method: str
    coverage: float
    successful_scenarios: int
    mean_utility: float | None
    instability: float | None
    stable_score: float | None
    mean_runtime_seconds: float | None


@dataclass(frozen=True, slots=True)
class StableStrategy:
    method: str
    macro_scores: dict[str, float]
    common_datasets: tuple[str, ...]
    summaries: tuple[DatasetMethodSummary, ...]
    paired_gains: dict[tuple[str, str, str], float]


def _validate_runs(runs: Sequence[HistoricalRun]) -> None:
    if not runs:
        raise ValueError("historical runs must not be empty")
    identities: set[tuple[str, str, str]] = set()
    for run in runs:
        identity = (run.dataset, run.scenario, run.method)
        if identity in identities:
            raise ValueError(f"duplicate historical execution: {identity}")
        identities.add(identity)
        if run.status == SUCCESS and run.utility is None:
            raise ValueError(f"successful execution has no utility: {identity}")


def build_stable_strategy(
    runs: Iterable[HistoricalRun],
    *,
    eligible_methods: Sequence[str],
    stability_penalty: float,
    minimum_coverage: float,
    minimum_successful_scenarios: int,
    minimum_common_datasets: int,
    registry_order: Sequence[str] | None = None,
) -> StableStrategy:
    """Build the paper's dataset-macro, stability-penalized default.

    Failed runs contribute to coverage but receive no synthetic utility.
    A dataset enters the common comparison set only when every eligible method
    satisfies the same coverage and success-count requirements.
    """

    materialized = tuple(runs)
    _validate_runs(materialized)
    methods = tuple(dict.fromkeys(eligible_methods))
    if not methods:
        raise ValueError("eligible_methods must not be empty")
    if stability_penalty < 0:
        raise ValueError("stability_penalty must be nonnegative")
    if not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in [0, 1]")
    if minimum_successful_scenarios < 1 or minimum_common_datasets < 1:
        raise ValueError("minimum counts must be positive")

    scenario_sets: dict[str, set[str]] = defaultdict(set)
    grouped: dict[tuple[str, str], list[HistoricalRun]] = defaultdict(list)
    for run in materialized:
        scenario_sets[run.dataset].add(run.scenario)
        grouped[run.dataset, run.method].append(run)

    summaries: list[DatasetMethodSummary] = []
    eligible_by_dataset: dict[str, set[str]] = defaultdict(set)
    for dataset in sorted(scenario_sets):
        scenario_count = len(scenario_sets[dataset])
        for method in methods:
            method_runs = grouped.get((dataset, method), [])
            successful = [
                run for run in method_runs
                if run.status == SUCCESS and run.utility is not None
            ]
            coverage = len(successful) / scenario_count
            utilities = [float(run.utility) for run in successful]
            runtimes = [float(run.runtime_seconds) for run in successful]
            mean_utility = fmean(utilities) if utilities else None
            instability = pstdev(utilities) if len(utilities) >= 2 else (
                0.0 if utilities else None
            )
            stable_score = (
                mean_utility - stability_penalty * instability
                if mean_utility is not None and instability is not None else None
            )
            summaries.append(DatasetMethodSummary(
                dataset=dataset,
                method=method,
                coverage=coverage,
                successful_scenarios=len(successful),
                mean_utility=mean_utility,
                instability=instability,
                stable_score=stable_score,
                mean_runtime_seconds=fmean(runtimes) if runtimes else None,
            ))
            if (
                coverage >= minimum_coverage
                and len(successful) >= minimum_successful_scenarios
            ):
                eligible_by_dataset[dataset].add(method)

    common_datasets = tuple(
        dataset for dataset in sorted(scenario_sets)
        if eligible_by_dataset[dataset] == set(methods)
    )
    if len(common_datasets) < minimum_common_datasets:
        raise ValueError(
            "insufficient history: "
            f"{len(common_datasets)} common datasets, "
            f"need {minimum_common_datasets}"
        )

    by_identity = {(item.dataset, item.method): item for item in summaries}
    macro_scores = {
        method: fmean(
            float(by_identity[dataset, method].stable_score)
            for dataset in common_datasets
        )
        for method in methods
    }
    runtime_tiebreak = {
        method: fmean(
            float(by_identity[dataset, method].mean_runtime_seconds)
            for dataset in common_datasets
        )
        for method in methods
    }
    order = {
        method: index
        for index, method in enumerate(registry_order or methods)
    }
    winner = min(
        methods,
        key=lambda method: (
            -macro_scores[method],
            runtime_tiebreak[method],
            order.get(method, len(order)),
            method,
        ),
    )

    successful_lookup = {
        (run.dataset, run.scenario, run.method): float(run.utility)
        for run in materialized
        if run.status == SUCCESS and run.utility is not None
    }
    paired_gains: dict[tuple[str, str, str], float] = {}
    for run in materialized:
        if run.method == winner or run.status != SUCCESS or run.utility is None:
            continue
        anchor_utility = successful_lookup.get((run.dataset, run.scenario, winner))
        if anchor_utility is not None:
            paired_gains[run.dataset, run.scenario, run.method] = (
                float(run.utility) - anchor_utility
            )

    return StableStrategy(
        method=winner,
        macro_scores=macro_scores,
        common_datasets=common_datasets,
        summaries=tuple(summaries),
        paired_gains=paired_gains,
    )
