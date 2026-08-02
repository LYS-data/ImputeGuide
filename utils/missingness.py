"""Reusable helpers for injecting synthetic missingness into numeric matrices.

支持三种缺失机制：MCAR（完全随机）、MAR（随机缺失）、MNAR（非随机缺失）。
参考：Little & Rubin, "Statistical Analysis with Missing Data", 3rd Edition, 2019.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy.special import expit as sigmoid

logger = logging.getLogger(__name__)

MissingMechanism = Literal["MCAR", "MAR", "MNAR"]


def inject_mcar_missing(
    X: np.ndarray,
    missing_rate: float,
    random_state: int,
    *,
    protect_full_row_col: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject MCAR (Missing Completely At Random) missingness.

    Each value has equal probability of being missing, independent of all
    observed and unobserved values.
    """

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D numeric array.")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1).")

    rng = np.random.default_rng(random_state)
    missing_mask = rng.random(X.shape) < missing_rate

    if protect_full_row_col and X.size > 0:
        for col in range(X.shape[1]):
            if missing_mask[:, col].all():
                missing_mask[rng.integers(0, X.shape[0]), col] = False
        for row in range(X.shape[0]):
            if missing_mask[row, :].all():
                missing_mask[row, rng.integers(0, X.shape[1])] = False

    X_missing = np.array(X, copy=True)
    X_missing[missing_mask] = np.nan
    return X_missing, missing_mask


def inject_mar_missing(
    X: np.ndarray,
    missing_rate: float,
    random_state: int,
    *,
    obs_column_ratio: float = 0.5,
    protect_full_row_col: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject MAR (Missing At Random) missingness.

    The probability of a value being missing depends on fully observed
    columns (the first `obs_column_ratio` fraction of features), but not
    on the missing values themselves.

    P(missing_{ij}) = sigmoid(β₀ + Σ_k β_k · X_{ik})
    where k indexes only the fully-observed columns.
    """

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D numeric array.")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1).")
    n, d = X.shape
    if d < 2:
        raise ValueError("MAR requires at least 2 features (need observed columns).")

    rng = np.random.default_rng(random_state)
    n_obs_cols = max(1, int(d * obs_column_ratio))
    # Columns 0..n_obs_cols-1 are fully observed (used for missingness model)
    # Columns n_obs_cols..d-1 may have missing values
    target_cols = list(range(n_obs_cols, d)) if n_obs_cols < d else list(range(d))

    # Standardize observed columns for stable logistic regression
    X_obs = X[:, :n_obs_cols]
    X_obs_mean = np.nanmean(X_obs, axis=0, keepdims=True)
    X_obs_std = np.nanstd(X_obs, axis=0, keepdims=True)
    X_obs_std[X_obs_std == 0] = 1.0
    X_obs_scaled = (X_obs - X_obs_mean) / X_obs_std
    X_obs_scaled = np.nan_to_num(X_obs_scaled, nan=0.0)

    # Random coefficients for the logistic model
    beta = rng.normal(0, 0.5, size=n_obs_cols)
    logits = X_obs_scaled @ beta

    # Adjust intercept to roughly match target missing rate
    intercept = _find_intercept(logits, missing_rate, rng)
    probs = sigmoid(intercept + logits)

    missing_mask = np.zeros((n, d), dtype=bool)
    for j in target_cols:
        mask_col = rng.random(n) < probs
        missing_mask[:, j] = mask_col

    if protect_full_row_col and X.size > 0:
        for col in range(d):
            if missing_mask[:, col].all():
                missing_mask[rng.integers(0, n), col] = False
        for row in range(n):
            if missing_mask[row, :].all():
                missing_mask[row, rng.integers(0, d)] = False

    X_missing = np.array(X, copy=True)
    X_missing[missing_mask] = np.nan
    return X_missing, missing_mask


def inject_mnar_missing(
    X: np.ndarray,
    missing_rate: float,
    random_state: int,
    *,
    mechanism: str = "self-censored",
    protect_full_row_col: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject MNAR (Missing Not At Random) missingness.

    The probability of a value being missing depends on the value itself
    (and potentially other unobserved factors).

    Two mechanisms supported:
    - "self-censored": values below a quantile threshold are more likely missing.
      P(missing_{ij}) = sigmoid(α · (threshold_j - X_{ij}))
    - "extremes": values far from the median (both tails) are more likely missing.
      P(missing_{ij}) = sigmoid(α · |X_{ij} - median_j| - threshold_j)
    """

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D numeric array.")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1).")
    n, d = X.shape

    rng = np.random.default_rng(random_state)
    missing_mask = np.zeros((n, d), dtype=bool)

    for j in range(d):
        col = X[:, j]
        col_median = float(np.nanmedian(col))
        col_std = float(np.nanstd(col))
        if col_std < 1e-8:
            col_std = 1.0

        if mechanism == "self-censored":
            # Lower values are more likely missing
            threshold = float(np.quantile(col, missing_rate * 1.5))
            deviation = (threshold - col) / col_std
            alpha = 2.0
            logits = alpha * deviation
        elif mechanism == "extremes":
            # Extreme values (both tails) are more likely missing
            deviation = np.abs(col - col_median) / col_std
            threshold = float(np.quantile(deviation, 1.0 - missing_rate))
            alpha = 2.0
            logits = alpha * (deviation - threshold)
        else:
            raise ValueError(f"Unknown MNAR mechanism: {mechanism}")

        intercept = _find_intercept(logits, missing_rate, rng)
        probs = sigmoid(intercept + logits)
        probs = np.clip(probs, 0.01, 0.99)
        missing_mask[:, j] = rng.random(n) < probs

    if protect_full_row_col and X.size > 0:
        for col in range(d):
            if missing_mask[:, col].all():
                missing_mask[rng.integers(0, n), col] = False
        for row in range(n):
            if missing_mask[row, :].all():
                missing_mask[row, rng.integers(0, d)] = False

    X_missing = np.array(X, copy=True)
    X_missing[missing_mask] = np.nan
    return X_missing, missing_mask


def inject_missingness(
    X: np.ndarray,
    missing_rate: float,
    random_state: int,
    *,
    mechanism: MissingMechanism = "MCAR",
    protect_full_row_col: bool = True,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    """统一入口：根据mechanism参数分发到对应的缺失注入函数。

    Args:
        X: 输入数据矩阵 (n, d)
        missing_rate: 缺失比例，[0, 1)
        random_state: 随机种子
        mechanism: "MCAR" | "MAR" | "MNAR"
        protect_full_row_col: 是否保护全行/全列缺失
        **kwargs: 传递给具体mechanism的额外参数

    Returns:
        (X_missing, missing_mask)
    """
    if mechanism == "MCAR":
        return inject_mcar_missing(X, missing_rate, random_state, protect_full_row_col=protect_full_row_col)
    elif mechanism == "MAR":
        return inject_mar_missing(X, missing_rate, random_state, protect_full_row_col=protect_full_row_col, **kwargs)  # type: ignore[arg-type]
    elif mechanism == "MNAR":
        return inject_mnar_missing(X, missing_rate, random_state, protect_full_row_col=protect_full_row_col, **kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown missing mechanism: {mechanism}. Use MCAR, MAR, or MNAR.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_intercept(
    logits: np.ndarray,
    target_rate: float,
    rng: np.random.Generator,
    max_iter: int = 50,
) -> float:
    """Binary search for intercept that matches target missing rate."""
    lo, hi = -10.0, 10.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        rate = float(np.mean(sigmoid(mid + logits)))
        if abs(rate - target_rate) < 0.005:
            return mid
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
