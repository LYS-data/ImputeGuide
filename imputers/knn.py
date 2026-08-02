"""KNN-based imputers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.impute import KNNImputer

from imputers.base import BaseImputer


class KNNImputerWrapper(BaseImputer):
    """Wrapper around sklearn KNN imputation with configurable weighting."""

    name = "knn"

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "nan_euclidean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self._imputer: KNNImputer | None = None

    def _fit(self, X: np.ndarray) -> None:
        self._imputer = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
        )
        self._imputer.fit(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._imputer is None:
            raise RuntimeError("KNN imputer backend is not fitted.")
        return self._imputer.transform(X)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "n_neighbors": self.n_neighbors,
                "weights": self.weights,
                "metric": self.metric,
            }
        )
        return params


class KNNIImputer(KNNImputerWrapper):
    """Single retained KNN imputer entry for the project.

    We keep a single registry-facing KNNI method to simplify the candidate
    method pool while still allowing the standard sklearn KNN imputation
    backend to be tuned through ``n_neighbors`` if needed.
    """

    name = "knni"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(weights="uniform", **kwargs)


class KNNUniformImputer(KNNImputerWrapper):
    name = "knn_uniform"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(weights="uniform", **kwargs)


class KNNDistanceImputer(KNNImputerWrapper):
    name = "knn_distance"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(weights="distance", **kwargs)
