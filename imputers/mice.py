"""Strict hyperimpute-backed MICE wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imputers.base import BaseImputer

try:
    from hyperimpute.plugins.imputers.plugin_mice import plugin as hyperimpute_mice_plugin

    HAS_HYPERIMPUTE_MICE = True
except Exception:  # pragma: no cover - optional dependency
    hyperimpute_mice_plugin = None
    HAS_HYPERIMPUTE_MICE = False


class MICEImputer(BaseImputer):
    """MICE wrapper using hyperimpute's original plugin implementation only."""

    name = "mice"

    def __init__(
        self,
        *,
        n_imputations: int = 1,
        max_iter: int = 100,
        tol: float = 1e-3,
        initial_strategy: int = 0,
        imputation_order: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not HAS_HYPERIMPUTE_MICE:
            raise ImportError(
                "mice requires hyperimpute (strict mode). Install `hyperimpute` to enable this method."
            )
        self.n_imputations = n_imputations
        self.max_iter = max_iter
        self.tol = tol
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self._model: Any | None = None

    def _fit(self, X: np.ndarray) -> None:
        random_state = 0 if self.random_state is None else int(self.random_state)
        self._model = hyperimpute_mice_plugin(
            n_imputations=self.n_imputations,
            max_iter=self.max_iter,
            tol=self.tol,
            initial_strategy=self.initial_strategy,
            imputation_order=self.imputation_order,
            random_state=random_state,
        )
        self._model.fit(pd.DataFrame(X))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MICE model is not fitted.")
        out = self._model.transform(pd.DataFrame(X))
        return np.asarray(out, dtype=float)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "n_imputations": self.n_imputations,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "initial_strategy": self.initial_strategy,
                "imputation_order": self.imputation_order,
                "strict_hyperimpute": True,
                "hyperimpute_available": HAS_HYPERIMPUTE_MICE,
            }
        )
        return params
