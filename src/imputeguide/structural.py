"""Paired, label-free structural scoring for ImputeGuide.

These functions consume cached validation slices. Imputers must be fitted on
the full target table beforehand; target labels and hidden values are never
accepted by this API.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score


Array = np.ndarray
Clusterer = Callable[[Array, int], Array]
Preprocessor = Callable[[Array], Array]


@dataclass(frozen=True, slots=True)
class ComponentMap:
    """Training-frozen clipping interval for one structural component."""

    lower: float
    upper: float

    def normalize(self, value: float) -> float:
        if not np.isfinite(value) or self.upper <= self.lower:
            raise ValueError("invalid component value or clipping interval")
        clipped = np.clip(value, self.lower, self.upper)
        return float((clipped - self.lower) / (self.upper - self.lower))


@dataclass(frozen=True, slots=True)
class ClusteringSeeds:
    full: int
    row_samples: tuple[int, ...]
    feature_subsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StructuralPerturbation:
    """One candidate-independent column in the paired evidence matrix."""

    identifier: str
    row_samples: tuple[tuple[int, ...], ...]
    feature_subsets: tuple[tuple[int, ...], ...]
    seeds: ClusteringSeeds


@dataclass(frozen=True, slots=True)
class PerturbationPlan:
    discovery: tuple[StructuralPerturbation, ...]
    confirmation: tuple[StructuralPerturbation, ...]


@dataclass(frozen=True, slots=True)
class StructuralComponents:
    silhouette: float
    row_consistency: float
    feature_agreement: float


@dataclass(frozen=True, slots=True)
class StructuralScore:
    score: float | None
    components: StructuralComponents | None
    valid: bool
    reason: str | None = None


def _sample_feature_groups(
    rng: np.random.Generator,
    feature_groups: Sequence[Sequence[int]],
    minimum_features: int,
    fraction: float,
) -> tuple[int, ...]:
    target_groups = max(1, int(np.ceil(fraction * len(feature_groups))))
    chosen: list[int] = []
    for offset, position in enumerate(rng.permutation(len(feature_groups))):
        chosen.extend(int(value) for value in feature_groups[int(position)])
        if len(chosen) >= minimum_features and offset + 1 >= target_groups:
            break
    return tuple(sorted(set(chosen)))


def build_perturbation_plan(
    *,
    n_rows: int,
    n_features: int,
    discovery_perturbations: int,
    confirmation_perturbations: int,
    row_resamples: int,
    feature_subsets: int,
    row_fraction: float,
    feature_fraction: float,
    minimum_row_overlap: int,
    minimum_features: int,
    discovery_seed: int,
    confirmation_seed: int,
    feature_groups: Sequence[Sequence[int]] | None = None,
) -> PerturbationPlan:
    """Freeze disjoint discovery and confirmation plans before scoring.

    ``feature_groups`` keeps encoded fragments of one source attribute
    together. With no supplied grouping, each matrix column is one group.
    """

    if n_rows < 3 or n_features < 1:
        raise ValueError("validation matrix is too small")
    if discovery_perturbations < 1 or confirmation_perturbations < 1:
        raise ValueError("both perturbation blocks must be nonempty")
    if row_resamples < 2 or feature_subsets < 1:
        raise ValueError("need at least two row samples and one feature subset")
    if not 0 < row_fraction <= 1 or not 0 < feature_fraction <= 1:
        raise ValueError("sampling fractions must be in (0, 1]")
    if discovery_seed == confirmation_seed:
        raise ValueError("discovery and confirmation seed namespaces must differ")

    groups = feature_groups or tuple((index,) for index in range(n_features))
    flattened = [int(value) for group in groups for value in group]
    if sorted(flattened) != list(range(n_features)) or len(set(flattened)) != n_features:
        raise ValueError("feature_groups must partition all columns exactly once")
    row_count = min(
        n_rows,
        max(minimum_row_overlap, int(np.ceil(row_fraction * n_rows))),
    )
    if row_count < 2:
        raise ValueError("row samples are too small")
    minimum_features = min(n_features, max(1, minimum_features))

    def block(prefix: str, count: int, seed: int) -> tuple[StructuralPerturbation, ...]:
        rng = np.random.default_rng(seed)
        output: list[StructuralPerturbation] = []
        for index in range(count):
            rows = tuple(
                tuple(sorted(int(value) for value in rng.choice(
                    n_rows, size=row_count, replace=False
                )))
                for _ in range(row_resamples)
            )
            features = tuple(
                _sample_feature_groups(rng, groups, minimum_features, feature_fraction)
                for _ in range(feature_subsets)
            )
            seeds = ClusteringSeeds(
                full=int(rng.integers(0, 2**31 - 1)),
                row_samples=tuple(int(value) for value in rng.integers(
                    0, 2**31 - 1, size=row_resamples
                )),
                feature_subsets=tuple(int(value) for value in rng.integers(
                    0, 2**31 - 1, size=feature_subsets
                )),
            )
            output.append(StructuralPerturbation(
                identifier=f"{prefix}-{index:03d}",
                row_samples=rows,
                feature_subsets=features,
                seeds=seeds,
            ))
        return tuple(output)

    return PerturbationPlan(
        discovery=block("discovery", discovery_perturbations, discovery_seed),
        confirmation=block("confirmation", confirmation_perturbations, confirmation_seed),
    )


def _labels(clusterer: Clusterer, matrix: Array, seed: int) -> Array:
    labels = np.asarray(clusterer(matrix, seed))
    if labels.ndim != 1 or len(labels) != len(matrix):
        raise ValueError("clusterer returned labels with an invalid shape")
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        raise ValueError("clusterer returned a degenerate partition")
    return labels


def score_perturbation(
    matrix: Array,
    perturbation: StructuralPerturbation,
    *,
    clusterer: Clusterer,
    component_maps: tuple[ComponentMap, ComponentMap, ComponentMap],
    weights: tuple[float, float, float],
    preprocessor: Preprocessor | None = None,
    minimum_row_overlap: int = 2,
) -> StructuralScore:
    """Score one cached completion under one shared perturbation."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        return StructuralScore(None, None, False, "nonfinite_or_nonmatrix_completion")
    if len(perturbation.row_samples) != len(perturbation.seeds.row_samples):
        return StructuralScore(None, None, False, "row_seed_count_mismatch")
    if len(perturbation.feature_subsets) != len(perturbation.seeds.feature_subsets):
        return StructuralScore(None, None, False, "feature_seed_count_mismatch")
    transform = preprocessor or (lambda array: array)
    try:
        prepared = np.asarray(transform(values), dtype=float)
        full_labels = _labels(clusterer, prepared, perturbation.seeds.full)
        raw_silhouette = float(silhouette_score(prepared, full_labels))

        sampled: list[tuple[Array, Array]] = []
        for indices, seed in zip(
            perturbation.row_samples, perturbation.seeds.row_samples
        ):
            rows = np.asarray(indices, dtype=int)
            if len(rows) < 3 or rows.min() < 0 or rows.max() >= len(values):
                raise ValueError("invalid row sample")
            labels = _labels(clusterer, np.asarray(transform(values[rows])), seed)
            sampled.append((rows, labels))
        agreements: list[float] = []
        for (left_rows, left_labels), (right_rows, right_labels) in combinations(sampled, 2):
            shared, left_positions, right_positions = np.intersect1d(
                left_rows, right_rows, assume_unique=True, return_indices=True
            )
            if len(shared) < minimum_row_overlap:
                raise ValueError("insufficient row-sample overlap")
            agreements.append(float(adjusted_rand_score(
                left_labels[left_positions], right_labels[right_positions]
            )))
        row_consistency = float(np.mean(agreements))

        feature_agreements: list[float] = []
        for indices, seed in zip(
            perturbation.feature_subsets, perturbation.seeds.feature_subsets
        ):
            columns = np.asarray(indices, dtype=int)
            if len(columns) < 1 or columns.min() < 0 or columns.max() >= values.shape[1]:
                raise ValueError("invalid feature subset")
            subset_labels = _labels(
                clusterer,
                np.asarray(transform(values[:, columns]), dtype=float),
                seed,
            )
            feature_agreements.append(float(adjusted_rand_score(
                full_labels, subset_labels
            )))
        feature_agreement = float(np.mean(feature_agreements))

        raw = (raw_silhouette, row_consistency, feature_agreement)
        normalized = tuple(
            mapping.normalize(value) for mapping, value in zip(component_maps, raw)
        )
        if len(weights) != 3 or any(weight < 0 for weight in weights):
            raise ValueError("structural weights must be three nonnegative values")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("structural weights must sum to one")
        components = StructuralComponents(*normalized)
        score = float(sum(weight * value for weight, value in zip(weights, normalized)))
        return StructuralScore(score, components, True)
    except (IndexError, TypeError, ValueError) as error:
        return StructuralScore(None, None, False, str(error))


def score_plan(
    matrix: Array,
    perturbations: Sequence[StructuralPerturbation],
    **score_kwargs: object,
) -> tuple[StructuralScore, ...]:
    """Score a matrix on a block without dropping invalid units."""

    return tuple(
        score_perturbation(matrix, perturbation, **score_kwargs)
        for perturbation in perturbations
    )
