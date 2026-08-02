"""Fail-safe final selection from frozen structural evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .confirmation import ConfirmationDecision, confirm_challenger


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_method: str | None
    challenger: str | None
    stop_reason: str
    attempted_methods: tuple[str, ...]
    successful_methods: tuple[str, ...]
    discovery_gains: dict[str, float]
    confirmation: ConfirmationDecision | None


def select_from_structural_evidence(
    *,
    anchor: str,
    candidates: Sequence[str],
    completion_success: Mapping[str, bool],
    discovery_scores: Mapping[str, np.ndarray],
    confirmation_scores: Mapping[str, np.ndarray],
    opportunity_score: float,
    opportunity_threshold: float,
    confirmation_alpha: float,
    bootstrap_repeats: int,
    switch_margin: float,
    bootstrap_seed: int,
    full_table_budget: int,
) -> SelectionResult:
    """Execute the selection rule over already-computed paired evidence.

    This pure function deliberately does not inspect labels or missing-value
    ground truth. Failed alternatives cannot invalidate a successful anchor.
    """

    unique_candidates = tuple(
        method for method in dict.fromkeys(candidates) if method != anchor
    )
    if full_table_budget < 1:
        raise ValueError("full_table_budget must reserve one anchor attempt")
    # The opportunity gate precedes all alternative full-table runs.
    gate_attempted = (anchor,)
    if not completion_success.get(anchor, False):
        return SelectionResult(
            selected_method=None,
            challenger=None,
            stop_reason="stable_strategy_failure",
            attempted_methods=gate_attempted,
            successful_methods=(),
            discovery_gains={},
            confirmation=None,
        )

    if opportunity_score <= opportunity_threshold or not unique_candidates:
        return SelectionResult(
            selected_method=anchor,
            challenger=None,
            stop_reason="opportunity_gate",
            attempted_methods=gate_attempted,
            successful_methods=gate_attempted,
            discovery_gains={},
            confirmation=None,
        )

    attempted = (anchor, *unique_candidates)
    if len(attempted) > full_table_budget:
        raise ValueError(
            f"{len(attempted)} full-table attempts exceed budget {full_table_budget}"
        )
    successful = tuple(
        method for method in attempted if completion_success.get(method, False)
    )

    if anchor not in discovery_scores:
        raise ValueError("anchor discovery scores are missing")
    anchor_discovery = np.asarray(discovery_scores[anchor], dtype=float)
    gains: dict[str, float] = {}
    for method in unique_candidates:
        if method not in successful or method not in discovery_scores:
            continue
        values = np.asarray(discovery_scores[method], dtype=float)
        if values.shape != anchor_discovery.shape:
            continue
        if not np.isfinite(values).all() or not np.isfinite(anchor_discovery).all():
            continue
        gains[method] = float((values - anchor_discovery).mean())

    positive = [method for method in unique_candidates if gains.get(method, 0.0) > 0]
    if not positive:
        return SelectionResult(
            selected_method=anchor,
            challenger=None,
            stop_reason="no_positive_discovery_gain",
            attempted_methods=attempted,
            successful_methods=successful,
            discovery_gains=gains,
            confirmation=None,
        )
    ranking_position = {method: index for index, method in enumerate(unique_candidates)}
    challenger = min(
        positive,
        key=lambda method: (-gains[method], ranking_position[method]),
    )
    if anchor not in confirmation_scores or challenger not in confirmation_scores:
        return SelectionResult(
            selected_method=anchor,
            challenger=challenger,
            stop_reason="invalid_confirmation",
            attempted_methods=attempted,
            successful_methods=successful,
            discovery_gains=gains,
            confirmation=None,
        )
    try:
        decision = confirm_challenger(
            anchor=anchor,
            challenger=challenger,
            anchor_scores=confirmation_scores[anchor],
            challenger_scores=confirmation_scores[challenger],
            alpha=confirmation_alpha,
            bootstrap_repeats=bootstrap_repeats,
            margin=switch_margin,
            seed=bootstrap_seed,
        )
    except ValueError:
        return SelectionResult(
            selected_method=anchor,
            challenger=challenger,
            stop_reason="invalid_confirmation",
            attempted_methods=attempted,
            successful_methods=successful,
            discovery_gains=gains,
            confirmation=None,
        )
    return SelectionResult(
        selected_method=decision.selected_method,
        challenger=challenger,
        stop_reason="switched" if decision.switched else "confirmation_gate",
        attempted_methods=attempted,
        successful_methods=successful,
        discovery_gains=gains,
        confirmation=decision,
    )
