"""Statistics helpers for current and future evaluation modules."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from utils.validation import ensure_numeric_matrix

logger = logging.getLogger(__name__)


def compute_missing_rate(X: np.ndarray | pd.DataFrame) -> float:
    """Return overall missing ratio."""
    array = ensure_numeric_matrix(X)
    return float(np.isnan(array).mean())


def compute_column_missing_rates(X: np.ndarray | pd.DataFrame) -> dict[int, float]:
    """Return per-column missing ratios."""
    array = ensure_numeric_matrix(X)
    return {idx: float(rate) for idx, rate in enumerate(np.isnan(array).mean(axis=0))}


def compare_basic_statistics(
    X_before: np.ndarray | pd.DataFrame,
    X_after: np.ndarray | pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Summarize how basic numeric moments changed after imputation."""
    before = ensure_numeric_matrix(X_before)
    after = ensure_numeric_matrix(X_after)
    return {
        "mean_abs_shift": _metric_delta(np.nanmean(before, axis=0), np.mean(after, axis=0)),
        "std_abs_shift": _metric_delta(np.nanstd(before, axis=0), np.std(after, axis=0)),
        "min_abs_shift": _metric_delta(np.nanmin(before, axis=0), np.min(after, axis=0)),
        "max_abs_shift": _metric_delta(np.nanmax(before, axis=0), np.max(after, axis=0)),
    }


def validate_imputed_result(X: np.ndarray | pd.DataFrame) -> dict[str, int | bool]:
    """Check whether an imputed result is fully numeric and finite."""
    array = ensure_numeric_matrix(X)
    return {
        "remaining_nan": int(np.isnan(array).sum()),
        "has_inf": bool(np.isinf(array).any()),
        "is_valid": bool((not np.isnan(array).any()) and (not np.isinf(array).any())),
    }


# ---------------------------------------------------------------------------
# Statistical significance tests for method comparison
# ---------------------------------------------------------------------------


def paired_wilcoxon_test(
    method_a_scores: list[float],
    method_b_scores: list[float],
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Wilcoxon signed-rank test for paired comparisons.

    Tests whether method A and method B have significantly different
    performance across multiple datasets/seeds.

    Args:
        method_a_scores: Scores from method A (one per dataset/seed).
        method_b_scores: Scores from method B (paired with A).
        alpha: Significance level.
        alternative: "two-sided", "greater" (A > B), or "less" (A < B).

    Returns:
        {"statistic": float, "p_value": float, "significant": bool,
         "median_diff": float, "mean_diff": float}
    """
    if len(method_a_scores) < 3:
        logger.warning("Wilcoxon test requires at least 3 paired observations (got %d).",
                       len(method_a_scores))
        return {
            "statistic": float("nan"), "p_value": float("nan"),
            "significant": False, "median_diff": float("nan"), "mean_diff": float("nan"),
        }

    a = np.asarray(method_a_scores, dtype=float)
    b = np.asarray(method_b_scores, dtype=float)

    # Remove NaN pairs
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]

    if len(a) < 3:
        return {
            "statistic": float("nan"), "p_value": float("nan"),
            "significant": False, "median_diff": float("nan"), "mean_diff": float("nan"),
        }

    result = scipy_stats.wilcoxon(a, b, alternative=alternative)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < alpha),
        "median_diff": float(np.median(a - b)),
        "mean_diff": float(np.mean(a - b)),
    }


def friedman_test(
    method_scores: dict[str, list[float]],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Friedman rank test for comparing multiple methods across datasets.

    Null hypothesis: all methods have the same performance distribution.

    Args:
        method_scores: {method_name: [scores across datasets/seeds]}.
        alpha: Significance level.

    Returns:
        {"statistic": float, "p_value": float, "significant": bool,
         "n_methods": int, "n_datasets": int}
    """
    methods = list(method_scores.keys())
    if len(methods) < 2:
        return {
            "statistic": float("nan"), "p_value": float("nan"),
            "significant": False, "n_methods": len(methods), "n_datasets": 0,
        }

    # Build matrix: rows = datasets, cols = methods
    arrays = [np.asarray(method_scores[m], dtype=float) for m in methods]
    min_len = min(len(a) for a in arrays)
    if min_len < 3:
        logger.warning("Friedman test requires at least 3 observations per method (got %d).",
                       min_len)
        return {
            "statistic": float("nan"), "p_value": float("nan"),
            "significant": False, "n_methods": len(methods), "n_datasets": min_len,
        }

    data = np.column_stack([a[:min_len] for a in arrays])
    # Remove rows with NaN
    valid_rows = np.all(np.isfinite(data), axis=1)
    data = data[valid_rows]

    if data.shape[0] < 3:
        return {
            "statistic": float("nan"), "p_value": float("nan"),
            "significant": False, "n_methods": len(methods), "n_datasets": data.shape[0],
        }

    try:
        statistic, p_value = scipy_stats.friedmanchisquare(*[data[:, j] for j in range(data.shape[1])])
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha),
            "n_methods": len(methods),
            "n_datasets": data.shape[0],
        }
    except Exception:
        logger.warning("Friedman test failed; returning NaN.", exc_info=True)
        return {
            "statistic": float("nan"), "p_value": float("nan"),
            "significant": False, "n_methods": len(methods), "n_datasets": data.shape[0],
        }


def compute_effect_size(
    method_a_scores: list[float],
    method_b_scores: list[float],
) -> dict[str, float]:
    """Compute Cohen's d and Cliff's delta effect sizes.

    Args:
        method_a_scores: Scores from method A.
        method_b_scores: Scores from method B.

    Returns:
        {"cohens_d": float, "cliffs_delta": float}
    """
    a = np.asarray(method_a_scores, dtype=float)
    b = np.asarray(method_b_scores, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]

    if len(a) < 3:
        return {"cohens_d": float("nan"), "cliffs_delta": float("nan")}

    # Cohen's d (paired)
    diff = a - b
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    cohens_d = float(mean_diff / std_diff) if std_diff > 1e-12 else 0.0

    # Cliff's delta (non-parametric effect size)
    # Counts proportion of a_i > b_j minus proportion of a_i < b_j
    n = len(a)
    greater = 0
    less = 0
    for i in range(n):
        for j in range(n):
            if a[i] > b[j]:
                greater += 1
            elif a[i] < b[j]:
                less += 1
    cliffs_delta = float((greater - less) / (n * n)) if n > 0 else float("nan")

    return {"cohens_d": cohens_d, "cliffs_delta": cliffs_delta}


def summarize_with_ci(
    scores: list[float],
    *,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compute mean with bootstrap confidence interval.

    Args:
        scores: List of scores.
        confidence: Confidence level (default 0.95).

    Returns:
        {"mean": float, "std": float, "ci_lower": float, "ci_upper": float, "n": int}
    """
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan"), "n": 0}

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))

    # Bootstrap CI
    rng = np.random.default_rng(42)
    n_bootstrap = 2000
    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1.0 - confidence) / 2.0
    ci_lower = float(np.percentile(boot_means, 100 * alpha))
    ci_upper = float(np.percentile(boot_means, 100 * (1.0 - alpha)))

    return {"mean": mean, "std": std, "ci_lower": ci_lower, "ci_upper": ci_upper, "n": len(arr)}


def _metric_delta(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    diff = np.abs(np.asarray(before, dtype=float) - np.asarray(after, dtype=float))
    return {
        "mean": float(np.mean(diff)),
        "max": float(np.max(diff)),
    }
