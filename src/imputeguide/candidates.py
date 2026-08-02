"""Deterministic quota-bounded dual-source candidate construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class CandidateSet:
    historical: tuple[str, ...]
    probe: tuple[str, ...]
    merged: tuple[str, ...]


def _unique_without_anchor(values: Iterable[str], anchor: str) -> list[str]:
    result: list[str] = []
    for value in values:
        if value != anchor and value not in result:
            result.append(value)
    return result


def merge_candidate_rankings(
    historical_ranking: Sequence[str],
    probe_ranking: Sequence[str],
    *,
    anchor: str,
    historical_quota: int,
    probe_quota: int,
) -> CandidateSet:
    """Alternately merge frozen rankings under independent branch quotas."""

    if historical_quota < 0 or probe_quota < 0:
        raise ValueError("candidate quotas must be nonnegative")
    history = _unique_without_anchor(historical_ranking, anchor)
    probe = _unique_without_anchor(probe_ranking, anchor)
    selected_history: list[str] = []
    selected_probe: list[str] = []
    merged: list[str] = []
    positions = [0, 0]
    rankings = (history, probe)
    selected = (selected_history, selected_probe)
    quotas = (historical_quota, probe_quota)
    while len(merged) < historical_quota + probe_quota:
        progress = False
        for branch in (0, 1):
            if len(selected[branch]) >= quotas[branch]:
                continue
            while positions[branch] < len(rankings[branch]):
                method = rankings[branch][positions[branch]]
                positions[branch] += 1
                if method in merged:
                    continue
                selected[branch].append(method)
                merged.append(method)
                progress = True
                break
        if not progress:
            break
    return CandidateSet(
        historical=tuple(selected_history),
        probe=tuple(selected_probe),
        merged=tuple(merged),
    )
