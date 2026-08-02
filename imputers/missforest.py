"""Strict hyperimpute-backed MissForest wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imputers.base import BaseImputer

try:
    from hyperimpute.plugins.imputers.plugin_missforest import plugin as hyperimpute_missforest_plugin
    from hyperimpute.plugins.imputers.plugin_hyperimpute import (
        plugin as hyperimpute_base_plugin,
    )

    HAS_HYPERIMPUTE_MISSFOREST = True
except Exception:  # pragma: no cover - optional dependency
    hyperimpute_missforest_plugin = None
    hyperimpute_base_plugin = None
    HAS_HYPERIMPUTE_MISSFOREST = False


class MissForestImputer(BaseImputer):
    """MissForest wrapper using hyperimpute's original plugin only."""

    name = "missforest"

    def __init__(
        self,
        *,
        n_estimators: int = 10,
        max_iter: int = 100,
        initial_strategy: int = 0,
        imputation_order: int = 0,
        force_single_thread: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not HAS_HYPERIMPUTE_MISSFOREST:
            raise ImportError(
                "missforest requires hyperimpute (strict mode). Install `hyperimpute` to enable this method."
            )
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.force_single_thread = force_single_thread
        self._model: Any | None = None

    @staticmethod
    def _patch_hyperimpute_random_forest_single_thread() -> None:
        """Patch hyperimpute RF plugins to avoid multiprocessing permission issues.

        Hyperimpute's RF plugins set n_jobs based on multiprocessing.cpu_count().
        In restricted Windows environments this can trigger a PermissionError when
        joblib initializes worker infrastructure. For strict-source execution we
        keep algorithm logic intact and only force cpu_count() to 1 inside the
        two RF plugin modules used by missforest.
        """
        try:
            import hyperimpute.plugins.prediction.classifiers.plugin_random_forest as rf_clf
            import hyperimpute.plugins.prediction.regression.plugin_random_forest_regressor as rf_reg

            rf_clf.multiprocessing.cpu_count = lambda: 1
            rf_reg.multiprocessing.cpu_count = lambda: 1
        except Exception:
            # If patching fails we keep original behavior and surface runtime errors.
            pass

    def _fit(self, X: np.ndarray) -> None:
        random_state = 0 if self.random_state is None else int(self.random_state)
        if self.force_single_thread:
            self._patch_hyperimpute_random_forest_single_thread()
        # Some hyperimpute versions expose fewer strategy/order values than others.
        # Clamp indices to avoid out-of-range crashes during automatic tuning.
        initial_vals = getattr(hyperimpute_base_plugin, "initial_strategy_vals", ["mean"])
        order_vals = getattr(
            hyperimpute_base_plugin,
            "imputation_order_vals",
            ["ascending", "descending", "roman", "arabic", "random"],
        )
        safe_initial_strategy = int(np.clip(self.initial_strategy, 0, max(len(initial_vals) - 1, 0)))
        safe_imputation_order = int(np.clip(self.imputation_order, 0, max(len(order_vals) - 1, 0)))
        self._model = hyperimpute_missforest_plugin(
            n_estimators=self.n_estimators,
            max_iter=self.max_iter,
            initial_strategy=safe_initial_strategy,
            imputation_order=safe_imputation_order,
            random_state=random_state,
        )
        self._model.fit(pd.DataFrame(X))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MissForest model is not fitted.")
        out = self._model.transform(pd.DataFrame(X))
        return np.asarray(out, dtype=float)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "n_estimators": self.n_estimators,
                "max_iter": self.max_iter,
                "initial_strategy": self.initial_strategy,
                "imputation_order": self.imputation_order,
                "force_single_thread": self.force_single_thread,
                "strict_hyperimpute": True,
                "hyperimpute_available": HAS_HYPERIMPUTE_MISSFOREST,
            }
        )
        return params
