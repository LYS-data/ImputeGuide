"""Held-out paired confirmation for a discovery-stage challenger."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ConfirmationDecision:
    selected_method: str
    switched: bool
    mean_gain: float
    gain_std: float
    empirical_lower_quantile: float
    margin: float
    paired_units: int
    bootstrap_repeats: int


def confirm_challenger(
    *,
    anchor: str,
    challenger: str,
    anchor_scores: np.ndarray,
    challenger_scores: np.ndarray,
    alpha: float,
    bootstrap_repeats: int,
    margin: float,
    seed: int,
    minimum_paired_units: int = 5,
) -> ConfirmationDecision:
    """Apply the paper's one-sided empirical paired-bootstrap switch gate."""

    anchor_values = np.asarray(anchor_scores, dtype=float)
    challenger_values = np.asarray(challenger_scores, dtype=float)
    if anchor_values.shape != challenger_values.shape:
        raise ValueError("anchor and challenger confirmation scores must be paired")
    if anchor_values.ndim != 1:
        raise ValueError("confirmation scores must be one-dimensional")
    if len(anchor_values) < minimum_paired_units:
        raise ValueError(
            f"at least {minimum_paired_units} paired confirmation units are required"
        )
    if not np.isfinite(anchor_values).all() or not np.isfinite(challenger_values).all():
        raise ValueError("all paired confirmation scores must be finite")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be a lower-tail probability in (0, 0.5)")
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")

    differences = challenger_values - anchor_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(differences), size=(bootstrap_repeats, len(differences))
    )
    means = differences[indices].mean(axis=1)
    lower = float(np.quantile(means, alpha))
    # Preserve the paper's strict gate under floating-point representation:
    # a value numerically equal to the margin must retain the anchor.
    equal_within_roundoff = np.isclose(lower, margin, rtol=1e-12, atol=1e-15)
    switched = bool(lower > margin and not equal_within_roundoff)
    return ConfirmationDecision(
        selected_method=challenger if switched else anchor,
        switched=switched,
        mean_gain=float(differences.mean()),
        gain_std=float(differences.std(ddof=1)),
        empirical_lower_quantile=lower,
        margin=float(margin),
        paired_units=len(differences),
        bootstrap_repeats=bootstrap_repeats,
    )
