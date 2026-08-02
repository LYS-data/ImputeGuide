"""GAIN wrapper for an optional local backend core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from imputers.base import BaseImputer

_LOCAL_CORE = (
    Path(__file__).resolve().parents[1]
    / "external_dependencies"
    / "gain_original"
    / "core.py"
)

try:
    if not _LOCAL_CORE.is_file():
        raise ImportError("optional GAIN backend core is not installed")
    from external_dependencies.gain_original import GAINCore

    HAS_TENSORFLOW = True
except Exception:  # pragma: no cover
    GAINCore = None
    HAS_TENSORFLOW = False


class GAINImputer(BaseImputer):
    """GAIN imputer with official core logic from jsyoon0823/GAIN.

    The wrapper preserves the original training objective and network
    architecture while adapting the interface to ``fit/transform`` on direct
    numeric matrices.
    """

    name = "gain"

    def __init__(
        self,
        *,
        batch_size: int = 128,
        n_epochs: int = 100,
        iterations: int | None = None,
        hint_rate: float = 0.9,
        loss_alpha: float = 100.0,
        **kwargs: Any,
    ) -> None:
        if not HAS_TENSORFLOW:
            raise ImportError("gain requires TensorFlow and its optional backend core.")
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.iterations = iterations if iterations is not None else n_epochs
        self.hint_rate = hint_rate
        self.loss_alpha = loss_alpha
        self._core: GAINCore | None = None

    def _fit(self, X: np.ndarray) -> None:
        self._core = GAINCore(
            batch_size=self.batch_size,
            hint_rate=self.hint_rate,
            alpha=self.loss_alpha,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        self._core.fit(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._core is None:
            raise RuntimeError("GAIN model is not fitted.")
        return self._core.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "iterations": self.iterations,
                "hint_rate": self.hint_rate,
                "loss_alpha": self.loss_alpha,
            }
        )
        return params
