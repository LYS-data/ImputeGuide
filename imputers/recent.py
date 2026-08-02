"""Wrappers for recent, peer-reviewed imputation baselines.

Both implementations are provided by the HyperImpute package.  The wrappers
only adapt pandas-based plugin APIs to the project's ``BaseImputer`` contract;
they do not replace either method with a local surrogate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import sklearn.linear_model as _sklearn_linear_model
from sklearn.linear_model import LogisticRegression as _SklearnLogisticRegression

from imputers.base import BaseImputer

try:
    from hyperimpute.plugins.imputers import Imputers

    _PLUGINS = Imputers()
    HAS_HYPERIMPUTE_RECENT = all(
        name in _PLUGINS.list() for name in ("hyperimpute", "miracle")
    )
except Exception:  # pragma: no cover - optional dependency
    Imputers = None
    _PLUGINS = None
    HAS_HYPERIMPUTE_RECENT = False


def _logistic_regression_compat(
    *args: Any,
    multi_class: str | None = None,
    **kwargs: Any,
) -> _SklearnLogisticRegression:
    """Build modern LogisticRegression while ignoring its removed option."""
    return _SklearnLogisticRegression(*args, **kwargs)


def _patch_hyperimpute_sklearn_compat() -> None:
    # HyperImpute dynamically loads prediction plugins during ``fit``.  Patch
    # the sklearn export as well as an already-imported plugin module so both
    # the lazy and eager paths receive the compatibility constructor.
    _sklearn_linear_model.LogisticRegression = _logistic_regression_compat
    try:
        from hyperimpute.plugins.prediction.classifiers import plugin_logistic_regression

        plugin_logistic_regression.LogisticRegression = _logistic_regression_compat
    except Exception:
        return


class _HyperImputePluginAdapter(BaseImputer):
    plugin_name: str

    def __init__(self, *, plugin_params: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not HAS_HYPERIMPUTE_RECENT or _PLUGINS is None:
            raise ImportError(
                f"{self.plugin_name} requires the HyperImpute package and its official plugin."
            )
        super().__init__(**kwargs)
        self.plugin_params = dict(plugin_params or {})
        self._model: Any | None = None

    def _fit(self, X: np.ndarray) -> None:
        _patch_hyperimpute_sklearn_compat()
        params = dict(self.plugin_params)
        params.setdefault("random_state", 0 if self.random_state is None else int(self.random_state))
        self._model = _PLUGINS.get(self.plugin_name, **params)
        self._model.fit(pd.DataFrame(X))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.plugin_name} model is not fitted.")
        return np.asarray(self._model.transform(pd.DataFrame(X)), dtype=float)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "plugin_name": self.plugin_name,
                "plugin_params": dict(self.plugin_params),
                "official_hyperimpute_plugin": True,
            }
        )
        return params


class HyperImputeImputer(_HyperImputePluginAdapter):
    """HyperImpute (ICML 2022) through its package plugin."""

    name = "hyperimpute"
    plugin_name = "hyperimpute"


class MiracleImputer(_HyperImputePluginAdapter):
    """MIRACLE (NeurIPS 2021) through its HyperImpute package plugin."""

    name = "miracle"
    plugin_name = "miracle"
