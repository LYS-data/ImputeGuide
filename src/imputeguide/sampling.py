"""Deterministic missingness-stratified row sampling."""

from __future__ import annotations

import numpy as np


def missing_fraction(observed_mask: np.ndarray) -> np.ndarray:
    """Return per-row missing fractions from a True-means-observed mask."""

    mask = np.asarray(observed_mask)
    if mask.ndim != 2 or mask.shape[1] < 1:
        raise ValueError("observed_mask must be a nonempty two-dimensional array")
    if mask.dtype != np.bool_:
        raise TypeError("observed_mask must have boolean dtype")
    return 1.0 - mask.mean(axis=1, dtype=float)


def stratified_validation_rows(
    observed_mask: np.ndarray,
    *,
    cap: int,
    training_quantile_cutpoints: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Draw proportionally within training-frozen missingness strata.

    Allocation uses largest remainders with deterministic stratum tie-breaking;
    row selection within each stratum uses the declared seed. If the table is
    no larger than ``cap``, every row is returned.
    """

    fractions = missing_fraction(observed_mask)
    n_rows = len(fractions)
    if cap < 1:
        raise ValueError("cap must be positive")
    if n_rows <= cap:
        return np.arange(n_rows, dtype=int)
    cutpoints = np.asarray(training_quantile_cutpoints, dtype=float)
    if cutpoints.ndim != 1 or not np.isfinite(cutpoints).all():
        raise ValueError("training cutpoints must be a finite vector")
    if len(cutpoints) and np.any(np.diff(cutpoints) <= 0):
        raise ValueError("training cutpoints must be strictly increasing")
    strata = np.digitize(fractions, cutpoints, right=True)
    identities, counts = np.unique(strata, return_counts=True)
    exact = cap * counts.astype(float) / n_rows
    allocations = np.floor(exact).astype(int)
    # Give every nonempty stratum one row when the cap permits it.
    if cap >= len(identities):
        allocations = np.maximum(allocations, 1)
    while allocations.sum() > cap:
        removable = np.flatnonzero(allocations > 1)
        position = min(
            removable,
            key=lambda index: (exact[index] - allocations[index], identities[index]),
        )
        allocations[position] -= 1
    while allocations.sum() < cap:
        eligible = np.flatnonzero(allocations < counts)
        position = max(
            eligible,
            key=lambda index: (exact[index] - allocations[index], -identities[index]),
        )
        allocations[position] += 1

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for identity, allocation in zip(identities, allocations):
        candidates = np.flatnonzero(strata == identity)
        selected.extend(int(value) for value in rng.choice(
            candidates, size=int(allocation), replace=False
        ))
    return np.asarray(sorted(selected), dtype=int)
