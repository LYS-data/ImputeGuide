"""Paper-aligned history and fixed-probe opportunity ranking."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ScenarioMatch:
    dataset: str
    scenario: str
    similarity: float


@dataclass(frozen=True, slots=True)
class HistoricalOpportunity:
    method: str
    score: float
    mean_gain: float
    weighted_std: float
    effective_sample_size: float
    paired_datasets: int
    maximum_similarity: float


@dataclass(frozen=True, slots=True)
class HistoricalRanking:
    supported: tuple[HistoricalOpportunity, ...]
    ranking: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProbeRanking:
    ranking: tuple[str, ...]
    representative_scores: dict[str, float]
    family_scores: dict[str, float]


def block_balanced_similarity(
    target_blocks: Sequence[np.ndarray],
    historical_blocks: Sequence[np.ndarray],
    *,
    block_weights: Sequence[float],
    epsilon: float,
) -> float:
    """Compute the paper's nonnegative weighted block-cosine similarity."""

    if len(target_blocks) != len(historical_blocks) or len(target_blocks) != len(block_weights):
        raise ValueError("profile block and weight counts must agree")
    if epsilon <= 0 or any(weight < 0 for weight in block_weights):
        raise ValueError("invalid similarity configuration")
    if not np.isclose(sum(block_weights), 1.0):
        raise ValueError("block weights must sum to one")
    total = 0.0
    for target, historical, weight in zip(
        target_blocks, historical_blocks, block_weights
    ):
        left = np.asarray(target, dtype=float).ravel()
        right = np.asarray(historical, dtype=float).ravel()
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("profile blocks must be aligned and finite")
        denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), epsilon)
        total += float(weight) * float(np.dot(left, right)) / denominator
    return max(0.0, float(total))


def rank_historical_opportunities(
    matches: Sequence[ScenarioMatch],
    paired_gains: Mapping[tuple[str, str, str], float],
    *,
    methods: Sequence[str],
    variation_penalty: float,
    maximum_scenarios_per_dataset: int,
    topmean_scenarios: int,
    minimum_paired_coverage: float,
    minimum_paired_datasets: int,
    minimum_effective_sample_size: float,
    minimum_maximum_similarity: float,
    registry_order: Sequence[str],
) -> HistoricalRanking:
    """Rank gains across distinct historical datasets without scenario bias."""

    if variation_penalty < 0:
        raise ValueError("variation_penalty must be nonnegative")
    if maximum_scenarios_per_dataset < 1 or topmean_scenarios < 1:
        raise ValueError("scenario caps must be positive")
    if not 0 <= minimum_paired_coverage <= 1:
        raise ValueError("minimum_paired_coverage must be in [0, 1]")
    positive = [match for match in matches if match.similarity > 0]
    by_dataset: dict[str, list[ScenarioMatch]] = {}
    for match in positive:
        by_dataset.setdefault(match.dataset, []).append(match)
    selected_by_dataset = {
        dataset: sorted(
            values, key=lambda item: (-item.similarity, item.scenario)
        )[:maximum_scenarios_per_dataset]
        for dataset, values in by_dataset.items()
    }
    dataset_weights = {
        dataset: float(np.mean([
            item.similarity for item in values[:topmean_scenarios]
        ]))
        for dataset, values in selected_by_dataset.items()
    }
    order = {method: index for index, method in enumerate(registry_order)}
    supported: list[HistoricalOpportunity] = []
    for method in dict.fromkeys(methods):
        dataset_gains: list[float] = []
        weights: list[float] = []
        similarities: list[float] = []
        for dataset, scenarios in selected_by_dataset.items():
            paired = [
                (item.similarity, paired_gains[dataset, item.scenario, method])
                for item in scenarios
                if (dataset, item.scenario, method) in paired_gains
            ]
            coverage = len(paired) / len(scenarios)
            if coverage < minimum_paired_coverage or not paired:
                continue
            scenario_weights = np.asarray([item[0] for item in paired], dtype=float)
            scenario_gains = np.asarray([item[1] for item in paired], dtype=float)
            dataset_gains.append(float(np.average(scenario_gains, weights=scenario_weights)))
            weights.append(dataset_weights[dataset])
            similarities.append(max(item.similarity for item in scenarios))
        if len(dataset_gains) < minimum_paired_datasets:
            continue
        weight_values = np.asarray(weights, dtype=float)
        gain_values = np.asarray(dataset_gains, dtype=float)
        effective = float(weight_values.sum() ** 2 / np.square(weight_values).sum())
        if effective < minimum_effective_sample_size:
            continue
        maximum_similarity = max(similarities)
        if maximum_similarity < minimum_maximum_similarity:
            continue
        mean = float(np.average(gain_values, weights=weight_values))
        variance = float(np.average(np.square(gain_values - mean), weights=weight_values))
        deviation = sqrt(max(0.0, variance))
        supported.append(HistoricalOpportunity(
            method=method,
            score=mean - variation_penalty * deviation,
            mean_gain=mean,
            weighted_std=deviation,
            effective_sample_size=effective,
            paired_datasets=len(dataset_gains),
            maximum_similarity=maximum_similarity,
        ))
    supported.sort(key=lambda item: (-item.score, order.get(item.method, len(order)), item.method))
    return HistoricalRanking(tuple(supported), tuple(item.method for item in supported))


def rank_probe_expansion(
    probe_scores: Mapping[str, np.ndarray],
    *,
    reference: str,
    representative_family: Mapping[str, str],
    within_family_ranking: Mapping[str, Sequence[str]],
    executable_methods: Sequence[str],
    anchor: str,
    variation_penalty: float,
    total_quota: int,
    per_family_cap: int,
    registry_order: Sequence[str],
) -> ProbeRanking:
    """Score fixed representatives, then expand families without extra probes."""

    if total_quota < 0 or per_family_cap < 1 or variation_penalty < 0:
        raise ValueError("invalid probe expansion configuration")
    if reference not in probe_scores:
        return ProbeRanking((), {}, {})
    reference_values = np.asarray(probe_scores[reference], dtype=float)
    if reference_values.ndim != 1 or not np.isfinite(reference_values).all():
        return ProbeRanking((), {}, {})
    representative_scores: dict[str, float] = {}
    family_scores: dict[str, float] = {}
    for representative, family in representative_family.items():
        if representative not in probe_scores:
            continue
        values = np.asarray(probe_scores[representative], dtype=float)
        if values.shape != reference_values.shape or not np.isfinite(values).all():
            continue
        differences = values - reference_values
        score = float(differences.mean() - variation_penalty * differences.std())
        representative_scores[representative] = score
        family_scores[family] = score
    order = {method: index for index, method in enumerate(registry_order)}
    executable = set(executable_methods)
    families = sorted(
        family_scores,
        key=lambda family: (-family_scores[family], family),
    )
    ranking: list[str] = []
    for family in families:
        members = sorted(
            dict.fromkeys(within_family_ranking.get(family, ())),
            key=lambda method: (
                list(within_family_ranking.get(family, ())).index(method),
                order.get(method, len(order)),
                method,
            ),
        )
        admitted = 0
        for method in members:
            if method == anchor or method not in executable or method in ranking:
                continue
            ranking.append(method)
            admitted += 1
            if admitted >= per_family_cap or len(ranking) >= total_quota:
                break
        if len(ranking) >= total_quota:
            break
    return ProbeRanking(tuple(ranking), representative_scores, family_scores)
