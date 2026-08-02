"""Public imputer exports, resolved lazily to avoid optional-stack startup."""

from importlib import import_module

__all__ = ["BaseImputer", "ImputerRegistry", "DEFAULT_REGISTRY", "build_imputer"]


_EXPORTS = {
    "BaseImputer": ("imputers.base", "BaseImputer"),
    "ImputerRegistry": ("imputers.registry", "ImputerRegistry"),
    "DEFAULT_REGISTRY": ("imputers.registry", "DEFAULT_REGISTRY"),
    "build_imputer": ("imputers.registry", "build_imputer"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
