"""Iterative imputers based on chained equations."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from imputers.base import BaseImputer

class IterativeEstimatorImputer(BaseImputer):
    """Generic sklearn iterative imputer wrapper."""

    name = "iterative"

    def __init__(
        self,
        *,
        estimator: Any,
        max_iter: int = 10,
        tol: float = 1e-3,
        sample_posterior: bool = False,
        imputation_order: str = "ascending",
        initial_strategy: str = "mean",
        skip_complete: bool = False,
        n_nearest_features: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.estimator = estimator
        self.max_iter = max_iter
        self.tol = tol
        self.sample_posterior = sample_posterior
        self.imputation_order = imputation_order
        self.initial_strategy = initial_strategy
        self.skip_complete = skip_complete
        self.n_nearest_features = n_nearest_features
        self._imputer: IterativeImputer | None = None

    def _fit(self, X: np.ndarray) -> None:
        self._imputer = IterativeImputer(
            estimator=self.estimator,
            max_iter=self.max_iter,
            tol=self.tol,
            sample_posterior=self.sample_posterior,
            imputation_order=self.imputation_order,
            initial_strategy=self.initial_strategy,
            skip_complete=self.skip_complete,
            n_nearest_features=self.n_nearest_features,
            random_state=self.random_state,
        )
        self._imputer.fit(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._imputer is None:
            raise RuntimeError("Iterative imputer backend is not fitted.")
        return self._imputer.transform(X)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "estimator": self.estimator.__class__.__name__,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "sample_posterior": self.sample_posterior,
                "imputation_order": self.imputation_order,
                "initial_strategy": self.initial_strategy,
                "skip_complete": self.skip_complete,
                "n_nearest_features": self.n_nearest_features,
            }
        )
        return params


class IterativeBayesianRidgeImputer(IterativeEstimatorImputer):
    name = "iterative_bayes_ridge"

    def __init__(self, **kwargs: Any) -> None:
        local_kwargs = dict(kwargs)
        random_state = local_kwargs.pop("random_state", None)
        estimator = BayesianRidge()
        super().__init__(estimator=estimator, random_state=random_state, **local_kwargs)


class IterativeExtraTreesImputer(IterativeEstimatorImputer):
    name = "iterative_extra_trees"

    def __init__(self, *, n_estimators: int = 50, **kwargs: Any) -> None:
        local_kwargs = dict(kwargs)
        random_state = local_kwargs.pop("random_state", None)
        estimator = ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=None,
        )
        super().__init__(estimator=estimator, random_state=random_state, **local_kwargs)
        self.n_estimators = n_estimators

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params["n_estimators"] = self.n_estimators
        return params


class IterativeRandomForestImputer(IterativeEstimatorImputer):
    name = "iterative_random_forest"

    def __init__(self, *, n_estimators: int = 50, **kwargs: Any) -> None:
        local_kwargs = dict(kwargs)
        random_state = local_kwargs.pop("random_state", None)
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=None,
        )
        super().__init__(estimator=estimator, random_state=random_state, **local_kwargs)
        self.n_estimators = n_estimators

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params["n_estimators"] = self.n_estimators
        return params
