"""Base abstractions for all imputation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from utils.validation import ensure_numeric_matrix, ensure_no_invalid_entries


class BaseImputer(ABC):
    """Unified interface for all numeric imputers."""

    name: str = "base"
    cost: float = 0.5

    def __init__(
        self,
        *,
        all_missing_policy: str = "raise",
        all_missing_fill_value: float = 0.0,
        random_state: int | None = None,
        cost: float | None = None,
    ) -> None:
        if all_missing_policy not in {"raise", "constant"}:
            raise ValueError("all_missing_policy must be either 'raise' or 'constant'.")
        self.all_missing_policy = all_missing_policy
        self.all_missing_fill_value = float(all_missing_fill_value)
        self.random_state = random_state
        if cost is not None:
            self.cost = float(cost)
        self._is_fitted = False
        self.n_features_in_: int | None = None
        self.all_missing_columns_: list[int] = []
        self.runtime_sec_: float | None = None

    def fit(self, X: np.ndarray | pd.DataFrame) -> "BaseImputer":
        X_array = ensure_numeric_matrix(X)
        X_prepared = self._prepare_input(X_array, fitting=True)
        self.n_features_in_ = X_prepared.shape[1]
        self._fit(X_prepared)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(f"{self.get_name()} must be fitted before transform().")
        X_array = ensure_numeric_matrix(X)
        if self.n_features_in_ is not None and X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"{self.get_name()} expected {self.n_features_in_} features, "
                f"but got {X_array.shape[1]}."
            )
        X_prepared = self._prepare_input(X_array, fitting=False)
        X_imputed = np.asarray(self._transform(X_prepared), dtype=float)
        ensure_no_invalid_entries(
            X_imputed,
            allow_nan=False,
            message=f"{self.get_name()} produced NaN or infinite values.",
        )
        return X_imputed

    def fit_transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        import time
        t0 = time.perf_counter()
        result = self.fit(X).transform(X)
        self.runtime_sec_ = time.perf_counter() - t0
        return result

    def get_name(self) -> str:
        return self.name

    def get_params(self) -> dict[str, Any]:
        return {
            "all_missing_policy": self.all_missing_policy,
            "all_missing_fill_value": self.all_missing_fill_value,
            "random_state": self.random_state,
        }

    def _prepare_input(self, X: np.ndarray, *, fitting: bool) -> np.ndarray:
        X_work = np.array(X, dtype=float, copy=True)
        all_missing = np.where(np.isnan(X_work).all(axis=0))[0].tolist()
        if fitting:
            self.all_missing_columns_ = all_missing
        if all_missing:
            if self.all_missing_policy == "raise":
                raise ValueError(
                    f"{self.get_name()} cannot fit data with fully-missing columns: "
                    f"{all_missing}. Use all_missing_policy='constant' to enable a "
                    "deterministic fallback."
                )
            X_work[:, all_missing] = self.all_missing_fill_value
        return X_work

    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        """Subclass-specific fitting implementation."""

    @abstractmethod
    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Subclass-specific transform implementation."""
