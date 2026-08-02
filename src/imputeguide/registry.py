"""Registry facade over the supported imputation-method adapters.

Backend imports are lazy so the core selector remains usable without loading
TensorFlow, PyTorch, graph extensions, or vendored research code.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


METHODS = (
    "mean",
    "median",
    "mode",
    "knni",
    "mice",
    "missforest",
    "iterative_rf",
    "soft_impute",
    "gain",
    "miwae",
    "hivae",
    "grape",
    "diffputer",
    "mcflow",
    "missdiff",
    "nomi",
    "remasker",
    "miri",
    "hyperimpute",
)

# Compatibility alias retained for earlier releases.
PAPER_METHODS = METHODS


def paper_methods() -> tuple[str, ...]:
    return METHODS


def methods() -> tuple[str, ...]:
    """Return the supported imputation methods in registry order."""

    return METHODS


def build_imputer(method: str, **parameters: Any) -> object:
    """Build a supported imputer adapter."""

    normalized = method.lower()
    if normalized not in METHODS:
        raise KeyError(f"unsupported imputation method: {method}")
    try:
        legacy_registry = import_module("imputers.registry")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "backend adapters are not installed; install requirements/full.txt"
        ) from error
    return legacy_registry.DEFAULT_REGISTRY.build(normalized, **parameters)


def build_paper_imputer(method: str, **parameters: Any) -> object:
    """Compatibility alias for :func:`build_imputer`."""

    return build_imputer(method, **parameters)


def describe_imputer(method: str) -> dict[str, Any]:
    """Describe a supported imputer and its dependencies."""

    normalized = method.lower()
    if normalized not in METHODS:
        raise KeyError(f"unsupported imputation method: {method}")
    legacy_registry = import_module("imputers.registry")
    return legacy_registry.DEFAULT_REGISTRY.describe(normalized)


def describe_paper_imputer(method: str) -> dict[str, Any]:
    """Compatibility alias for :func:`describe_imputer`."""

    return describe_imputer(method)
