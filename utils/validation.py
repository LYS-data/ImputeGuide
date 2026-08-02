"""Validation utilities shared across the project."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def ensure_numeric_matrix(X: np.ndarray | pd.DataFrame | Any) -> np.ndarray:
    """Convert supported tabular input into a 2D float array."""
    if isinstance(X, pd.DataFrame):
        array = X.to_numpy()
    else:
        array = np.asarray(X)

    if array.ndim != 2:
        raise ValueError("Input must be a 2D numeric matrix.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Input matrix must have at least one row and one column.")
    if not np.issubdtype(array.dtype, np.number):
        try:
            array = array.astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError("Input must contain only numeric values.") from exc
    return array.astype(float, copy=False)


def ensure_no_invalid_entries(
    X: np.ndarray,
    *,
    allow_nan: bool = False,
    message: str = "Array contains invalid entries.",
) -> None:
    """Raise when an array contains NaN or inf entries."""
    if not allow_nan and np.isnan(X).any():
        raise ValueError(message)
    if np.isinf(X).any():
        raise ValueError(message)


def get_all_missing_columns(X: np.ndarray | pd.DataFrame) -> list[int]:
    """Return column indices that are fully missing."""
    array = ensure_numeric_matrix(X)
    return np.where(np.isnan(array).all(axis=0))[0].tolist()
