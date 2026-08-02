"""Whole-table execution contract shared by every imputer adapter."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np


Builder = Callable[[], object]


@dataclass(frozen=True, slots=True)
class MethodAttempt:
    method: str
    status: str
    runtime_seconds: float
    completion: np.ndarray | None
    failure_type: str | None
    failure_message: str | None


def validate_completion(
    incomplete: np.ndarray,
    completion: np.ndarray,
    *,
    observed_tolerance: float = 0.0,
) -> np.ndarray:
    """Enforce the reusable whole-table completion contract."""

    source = np.asarray(incomplete, dtype=float)
    result = np.asarray(completion, dtype=float)
    if source.ndim != 2:
        raise ValueError("incomplete input must be a two-dimensional matrix")
    if result.shape != source.shape:
        raise ValueError(
            f"completion shape {result.shape} does not match input {source.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("completion contains NaN or infinite values")
    observed = np.isfinite(source)
    if not np.allclose(
        result[observed], source[observed], rtol=0.0, atol=observed_tolerance
    ):
        raise ValueError("completion changed observed cells")
    return np.array(result, copy=True)


def execute_whole_table(
    method: str,
    incomplete: np.ndarray,
    *,
    builder: Builder,
    observed_tolerance: float = 0.0,
) -> MethodAttempt:
    """Execute one adapter once and convert every failure into a trace record.

    Hard timeouts must be enforced by the outer process runner so timed-out GPU
    or native-library work can be terminated safely.
    """

    started = perf_counter()
    try:
        imputer = builder()
        transform = getattr(imputer, "fit_transform", None)
        if not callable(transform):
            raise TypeError("imputer adapter must expose fit_transform(matrix)")
        completion = validate_completion(
            incomplete,
            transform(np.array(incomplete, dtype=float, copy=True)),
            observed_tolerance=observed_tolerance,
        )
        return MethodAttempt(
            method=method,
            status="success",
            runtime_seconds=perf_counter() - started,
            completion=completion,
            failure_type=None,
            failure_message=None,
        )
    except Exception as error:  # failure is data, not a selector-wide crash
        return MethodAttempt(
            method=method,
            status="failure",
            runtime_seconds=perf_counter() - started,
            completion=None,
            failure_type=type(error).__name__,
            failure_message=str(error),
        )
