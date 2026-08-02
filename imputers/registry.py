"""Backend registry for the 19 ImputeGuide imputers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from imputers.base import BaseImputer
from imputers.diffputer import DiffPuterImputer, HAS_TORCH as HAS_DIFFPUTER
from imputers.gain import GAINImputer, HAS_TENSORFLOW as HAS_GAIN
from imputers.grape import GRAPEImputer, HAS_GRAPE_DEPS
from imputers.hivae import HIVAEImputer, HAS_TENSORFLOW as HAS_HIVAE
from imputers.iterative import IterativeRandomForestImputer
from imputers.knn import KNNIImputer
from imputers.matrix_factorization import HAS_FANCYIMPUTE, SoftImputeImputer
from imputers.mcflow import HAS_MCFLOW_DEPS, MCFlowImputer
from imputers.mice import HAS_HYPERIMPUTE_MICE, MICEImputer
from imputers.miri import HAS_MIRI_DEPS, MIRIImputer
from imputers.missdiff import HAS_MISSDIFF_DEPS, MissDiffImputer
from imputers.missforest import HAS_HYPERIMPUTE_MISSFOREST, MissForestImputer
from imputers.miwae import HAS_HYPERIMPUTE_MIWAE, MIWAEImputer
from imputers.nomi import HAS_NOMI_DEPS, NOMIImputer
from imputers.recent import HAS_HYPERIMPUTE_RECENT, HyperImputeImputer
from imputers.remasker import HAS_REMASKER_DEPS, ReMaskerImputer
from imputers.simple import (
    HAS_HYPERIMPUTE_SIMPLE,
    MeanImputer,
    MedianImputer,
    MostFrequentImputer,
)


@dataclass(slots=True)
class ImputerSpec:
    name: str
    builder: Callable[..., BaseImputer]
    default_params: dict[str, Any] = field(default_factory=dict)
    available: bool = True
    requires: tuple[str, ...] = ()
    notes: str | None = None


def _specs() -> tuple[ImputerSpec, ...]:
    return (
        ImputerSpec("mean", MeanImputer, available=HAS_HYPERIMPUTE_SIMPLE, requires=("hyperimpute",)),
        ImputerSpec("median", MedianImputer, available=HAS_HYPERIMPUTE_SIMPLE, requires=("hyperimpute",)),
        ImputerSpec("mode", MostFrequentImputer),
        ImputerSpec("knni", KNNIImputer, {"n_neighbors": 5}),
        ImputerSpec("mice", MICEImputer, {
            "n_imputations": 1, "max_iter": 100, "tol": 1e-3,
            "initial_strategy": 0, "imputation_order": 0,
        }, HAS_HYPERIMPUTE_MICE, ("hyperimpute",)),
        ImputerSpec("missforest", MissForestImputer, {
            "n_estimators": 10, "max_iter": 100, "initial_strategy": 0,
            "imputation_order": 0, "force_single_thread": True,
        }, HAS_HYPERIMPUTE_MISSFOREST, ("hyperimpute",)),
        ImputerSpec("iterative_rf", IterativeRandomForestImputer, {
            "max_iter": 10, "n_estimators": 100, "random_state": 42,
        }),
        ImputerSpec("soft_impute", SoftImputeImputer, {
            "max_rank": None, "max_iters": 100,
        }, HAS_FANCYIMPUTE, ("fancyimpute",)),
        ImputerSpec("gain", GAINImputer, {
            "batch_size": 64, "n_epochs": 100, "hint_rate": 0.9,
            "loss_alpha": 100.0,
        }, HAS_GAIN, ("tensorflow",)),
        ImputerSpec("miwae", MIWAEImputer, {
            "n_epochs": 500, "batch_size": 256, "latent_size": 1,
            "n_hidden": 1, "K": 20,
        }, HAS_HYPERIMPUTE_MIWAE, ("hyperimpute",)),
        ImputerSpec("hivae", HIVAEImputer, {
            "dim_latent_z": 2, "dim_latent_y": 3, "dim_latent_s": 4,
            "batch_size": 128, "epochs": 50,
        }, HAS_HIVAE, ("tensorflow",)),
        ImputerSpec("grape", GRAPEImputer, {
            "hidden_dim": 64, "max_epochs": 100, "learning_rate": 1e-3,
        }, HAS_GRAPE_DEPS, ("torch", "torch_geometric")),
        ImputerSpec("diffputer", DiffPuterImputer, {
            "hid_dim": 128, "max_iter": 1, "num_steps": 10,
            "num_trials": 2, "max_epochs": 50, "batch_size": 128,
        }, HAS_DIFFPUTER, ("torch",)),
        ImputerSpec("mcflow", MCFlowImputer, {
            "n_epochs": 20, "batch_size": 256, "num_nf_layers": 3,
            "learning_rate": 1e-4, "reset_imputations": True, "device": "cpu",
        }, HAS_MCFLOW_DEPS, ("torch",)),
        ImputerSpec("missdiff", MissDiffImputer, {
            "hidden_dim": 128, "max_epochs": 20, "batch_size": 1024,
            "learning_rate": 1e-4, "num_steps": 20, "num_trials": 1,
            "inference_batch_size": 2048,
        }, HAS_MISSDIFF_DEPS, ("torch",)),
        ImputerSpec("nomi", NOMIImputer, {
            "k_neighbors": 10, "max_iterations": 3, "tau": 1.0, "beta": 1.0,
        }, HAS_NOMI_DEPS, ("tensorflow", "hnswlib", "neural_tangents")),
        ImputerSpec("remasker", ReMaskerImputer, {
            "max_epochs": 20, "batch_size": 256, "mask_ratio": 0.5,
            "embed_dim": 32, "encoder_depth": 4, "decoder_depth": 2,
        }, HAS_REMASKER_DEPS, ("torch", "timm", "torchvision")),
        ImputerSpec("miri", MIRIImputer, {
            "max_rounds": 3, "max_epochs": 10, "ode_steps": 20,
            "batch_size": 256,
        }, HAS_MIRI_DEPS, ("torch",)),
        ImputerSpec("hyperimpute", HyperImputeImputer,
                    available=HAS_HYPERIMPUTE_RECENT, requires=("hyperimpute",)),
    )


class ImputerRegistry:
    def __init__(self) -> None:
        specs = _specs()
        self._registry = {spec.name: spec for spec in specs}
        if len(self._registry) != 19:
            raise RuntimeError("registry must contain exactly 19 unique methods")

    def list_imputers(self, *, available_only: bool = False) -> list[str]:
        methods = list(self._registry)
        return (
            [method for method in methods if self._registry[method].available]
            if available_only else methods
        )

    def get_spec(self, name: str) -> ImputerSpec:
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"Unknown imputer: {name}")
        return self._registry[key]

    def is_available(self, name: str) -> bool:
        return self.get_spec(name).available

    def get_default_params(self, name: str) -> dict[str, Any]:
        return dict(self.get_spec(name).default_params)

    def describe(self, name: str) -> dict[str, Any]:
        spec = self.get_spec(name)
        return {
            "name": spec.name,
            "available": spec.available,
            "requires": list(spec.requires),
            "default_params": dict(spec.default_params),
            "notes": spec.notes,
        }

    def build(self, name: str, **kwargs: Any) -> BaseImputer:
        spec = self.get_spec(name)
        if not spec.available:
            requirements = ", ".join(spec.requires) or "optional dependencies"
            raise ImportError(f"Imputer '{name}' is unavailable: missing {requirements}.")
        return spec.builder(**{**spec.default_params, **kwargs})


DEFAULT_REGISTRY = ImputerRegistry()


def build_imputer(name: str, **kwargs: Any) -> BaseImputer:
    return DEFAULT_REGISTRY.build(name, **kwargs)
