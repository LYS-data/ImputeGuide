"""NOMI wrapper for an optional local backend core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from packaging.version import Version

from imputers.base import BaseImputer

try:
    import hnswlib  # noqa: F401

    HAS_HNSWLIB = True
except ImportError:  # pragma: no cover
    HAS_HNSWLIB = False

_LOCAL_CORE = (
    Path(__file__).resolve().parents[1]
    / "external_dependencies"
    / "nomi_original"
    / "core.py"
)

try:
    import tensorflow as tf  # noqa: F401

    if Version(tf.__version__) >= Version("2.16") and _LOCAL_CORE.is_file():
        import neural_tangents  # noqa: F401
        from external_dependencies.nomi_original import NOMICore

        HAS_NEURAL_TANGENTS = True
    else:  # pragma: no cover
        NOMICore = None
        HAS_NEURAL_TANGENTS = False
except Exception:  # pragma: no cover
    NOMICore = None
    HAS_NEURAL_TANGENTS = False

HAS_NOMI_DEPS = HAS_HNSWLIB and HAS_NEURAL_TANGENTS


class NOMIImputer(BaseImputer):
    """NOMI wrapper with official GitHub core logic from guaiyoui/NOMI."""

    name = "nomi"

    def __init__(
        self,
        *,
        k_neighbors: int = 10,
        similarity_metric: str = "l2",
        max_iterations: int = 3,
        tau: float = 1.0,
        beta: float = 1.0,
        batch_cap: int = 300,
        diag_reg: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        if not HAS_NOMI_DEPS:
            raise ImportError(
                "nomi requires TensorFlow, hnswlib, neural-tangents, and its "
                "optional backend core."
            )
        super().__init__(**kwargs)
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.max_iterations = max_iterations
        self.tau = tau
        self.beta = beta
        self.batch_cap = batch_cap
        self.diag_reg = diag_reg
        self._core: NOMICore | None = None

    def _fit(self, X):
        self._core = NOMICore(
            k_neighbors=self.k_neighbors,
            similarity_metric=self.similarity_metric,
            max_iterations=self.max_iterations,
            tau=self.tau,
            beta=self.beta,
            batch_cap=self.batch_cap,
            diag_reg=self.diag_reg,
            random_state=self.random_state,
        )
        self._core.fit(X)

    def _transform(self, X):
        if self._core is None:
            raise RuntimeError("NOMI model is not fitted.")
        return self._core.transform(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "k_neighbors": self.k_neighbors,
                "similarity_metric": self.similarity_metric,
                "max_iterations": self.max_iterations,
                "tau": self.tau,
                "beta": self.beta,
                "batch_cap": self.batch_cap,
                "diag_reg": self.diag_reg,
            }
        )
        return params
