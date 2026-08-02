"""Strict hyperimpute-backed MIWAE wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imputers.base import BaseImputer

try:
    from hyperimpute.plugins.imputers.plugin_miwae import plugin as hyperimpute_miwae_plugin

    HAS_HYPERIMPUTE_MIWAE = True
except Exception:  # pragma: no cover - optional dependency
    hyperimpute_miwae_plugin = None
    HAS_HYPERIMPUTE_MIWAE = False

# Keep compatibility with existing tests/imports that refer to HAS_TORCH.
HAS_TORCH = HAS_HYPERIMPUTE_MIWAE


class MIWAEImputer(BaseImputer):
    """MIWAE wrapper using hyperimpute's original plugin only."""

    name = "miwae"

    def __init__(
        self,
        *,
        n_epochs: int = 500,
        batch_size: int = 256,
        latent_size: int = 1,
        n_hidden: int = 1,
        K: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not HAS_HYPERIMPUTE_MIWAE:
            raise ImportError("miwae requires hyperimpute (strict mode). Install `hyperimpute`.")
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.n_hidden = n_hidden
        self.K = K
        self._model: Any | None = None

    def _fit(self, X: np.ndarray) -> None:
        random_state = 0 if self.random_state is None else int(self.random_state)
        self._model = hyperimpute_miwae_plugin(
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            latent_size=self.latent_size,
            n_hidden=self.n_hidden,
            random_state=random_state,
            K=self.K,
        )
        self._model.fit(pd.DataFrame(X))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MIWAE model is not fitted.")
        out = self._model.transform(pd.DataFrame(X))
        return np.asarray(out, dtype=float)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "latent_size": self.latent_size,
                "n_hidden": self.n_hidden,
                "K": self.K,
                "strict_hyperimpute": True,
                "hyperimpute_available": HAS_HYPERIMPUTE_MIWAE,
            }
        )
        return params
