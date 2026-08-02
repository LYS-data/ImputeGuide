"""Low-rank and matrix factorization style imputers."""

from __future__ import annotations

from typing import Any

import numpy as np

from imputers.base import BaseImputer

try:
    from fancyimpute import SoftImpute
    from fancyimpute import solver as _fancy_solver
    from inspect import signature
    from fancyimpute import soft_impute as _fancy_soft_impute

    if "force_all_finite" not in signature(_fancy_solver.check_array).parameters:
        _original_check_array = _fancy_solver.check_array

        def _check_array_compat(*args: Any, force_all_finite: Any = None, **kwargs: Any):
            if force_all_finite is not None:
                kwargs.setdefault("ensure_all_finite", force_all_finite)
            return _original_check_array(*args, **kwargs)

        _fancy_solver.check_array = _check_array_compat
        _fancy_soft_impute.check_array = _check_array_compat

    HAS_FANCYIMPUTE = True
except ImportError:  # pragma: no cover
    SoftImpute = None
    HAS_FANCYIMPUTE = False


class SVDLowRankImputer(BaseImputer):
    """Approximate low-rank matrix completion using repeated truncated SVD."""

    name = "svd_low_rank"

    def __init__(
        self,
        *,
        rank: int = 2,
        max_iter: int = 20,
        tol: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self._column_means: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        self._column_means = np.nanmean(X, axis=0)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._column_means is None:
            raise RuntimeError("Low-rank imputer is not fitted.")
        missing_mask = np.isnan(X)
        X_filled = np.array(X, dtype=float, copy=True)
        for col_idx in range(X_filled.shape[1]):
            X_filled[missing_mask[:, col_idx], col_idx] = self._column_means[col_idx]

        previous = X_filled.copy()
        rank = max(1, min(self.rank, min(X_filled.shape) - 1))
        for _ in range(self.max_iter):
            U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
            reconstruction = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
            X_filled[missing_mask] = reconstruction[missing_mask]
            delta = np.linalg.norm(X_filled - previous) / (np.linalg.norm(previous) + 1e-12)
            if delta < self.tol:
                break
            previous = X_filled.copy()
        return X_filled

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update({"rank": self.rank, "max_iter": self.max_iter, "tol": self.tol})
        return params


class SoftImputeImputer(BaseImputer):
    """Optional fancyimpute wrapper kept explicit for dependency visibility."""

    name = "soft_impute"

    def __init__(
        self,
        *,
        shrinkage_value: float | None = None,
        max_rank: int | None = None,
        max_iters: int = 100,
        **kwargs: Any,
    ) -> None:
        if not HAS_FANCYIMPUTE:
            raise ImportError("soft_impute requires the optional dependency 'fancyimpute'.")
        super().__init__(**kwargs)
        self.shrinkage_value = shrinkage_value
        self.max_rank = max_rank
        self.max_iters = max_iters
        self._imputer: SoftImpute | None = None

    def _fit(self, X: np.ndarray) -> None:
        self._imputer = SoftImpute(
            shrinkage_value=self.shrinkage_value,
            max_rank=self.max_rank,
            max_iters=self.max_iters,
            init_fill_method="mean",
        )

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._imputer is None:
            raise RuntimeError("SoftImpute backend is not fitted.")
        return np.asarray(self._imputer.fit_transform(X), dtype=float)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "shrinkage_value": self.shrinkage_value,
                "max_rank": self.max_rank,
                "max_iters": self.max_iters,
            }
        )
        return params
