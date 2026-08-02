"""GRAPE wrapper with direct original training-flow connection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from imputers.base import BaseImputer

try:
    import torch
    HAS_GRAPE_DEPS = True
except Exception:  # pragma: no cover
    torch = None
    HAS_GRAPE_DEPS = False

get_data_fix_mask = None
out_of_sample_test_gnn_mdi = None
train_gnn_mdi = None


def _load_grape_flow() -> None:
    """Load the official graph stack only when GRAPE is actually selected."""
    global get_data_fix_mask, out_of_sample_test_gnn_mdi, train_gnn_mdi
    if get_data_fix_mask is not None:
        return
    import torch_geometric  # noqa: F401
    from external_dependencies.grape_original import (
        get_data_fix_mask as load_data,
        out_of_sample_test_gnn_mdi as out_of_sample,
        train_gnn_mdi as train,
    )
    get_data_fix_mask = load_data
    out_of_sample_test_gnn_mdi = out_of_sample
    train_gnn_mdi = train


class GRAPEImputer(BaseImputer):
    """Direct GRAPE training/inference wrapper.

    Design goal:
    - Keep the original GRAPE training flow unchanged.
    - Only do minimal interface adaptation for dense matrix input.
    """

    name = "grape"

    def __init__(
        self,
        *,
        hidden_dim: int | None = None,
        model_types: str = "EGSAGE_EGSAGE_EGSAGE",
        post_hiddens: str | None = None,
        concat_states: bool = False,
        norm_embs: str | None = None,
        aggr: str = "mean",
        node_dim: int = 64,
        edge_dim: int = 64,
        edge_mode: int = 1,
        gnn_activation: str = "relu",
        impute_hiddens: str = "64",
        impute_activation: str = "relu",
        max_epochs: int = 400,
        opt: str = "adam",
        opt_scheduler: str = "none",
        opt_decay_step: int = 1000,
        opt_decay_rate: float = 0.9,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        learning_rate: float = 0.001,
        known: float = 0.7,
        loss_mode: int = 0,
        node_mode: int = 0,
        early_stopping_patience: int = 20,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not HAS_GRAPE_DEPS:
            raise ImportError(
                "grape requires the official GRAPE dependency stack: "
                "torch and torch_geometric."
            )
        try:
            _load_grape_flow()
        except Exception as exc:
            raise ImportError(
                "GRAPE's official torch-geometric flow could not be loaded."
            ) from exc
        super().__init__(**kwargs)
        self.model_types = model_types
        self.post_hiddens = post_hiddens
        self.concat_states = concat_states
        self.norm_embs = norm_embs
        self.aggr = aggr
        # Backward-compatible alias used by existing configs/tests.
        if hidden_dim is not None:
            node_dim = hidden_dim
            edge_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.edge_mode = edge_mode
        self.gnn_activation = gnn_activation
        self.impute_hiddens = impute_hiddens
        self.impute_activation = impute_activation
        self.max_epochs = max_epochs
        self.opt = opt
        self.opt_scheduler = opt_scheduler
        self.opt_decay_step = opt_decay_step
        self.opt_decay_rate = opt_decay_rate
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.known = known
        self.loss_mode = loss_mode
        self.node_mode = node_mode
        self.early_stopping_patience = early_stopping_patience
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._impute_model = None
        self._fit_imputed_: np.ndarray | None = None

    def _build_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            model_types=self.model_types,
            post_hiddens=self.post_hiddens,
            concat_states=self.concat_states,
            norm_embs=self.norm_embs,
            aggr=self.aggr,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            edge_mode=self.edge_mode,
            gnn_activation=self.gnn_activation,
            impute_hiddens=self.impute_hiddens,
            impute_activation=self.impute_activation,
            epochs=self.max_epochs,
            opt=self.opt,
            opt_scheduler=self.opt_scheduler,
            opt_restart=0,
            opt_decay_step=self.opt_decay_step,
            opt_decay_rate=self.opt_decay_rate,
            dropout=self.dropout,
            weight_decay=self.weight_decay,
            lr=self.learning_rate,
            known=self.known,
            auto_known=False,
            loss_mode=self.loss_mode,
            valid=0.0,
            seed=0 if self.random_state is None else self.random_state,
            log_dir="",
            save_model=False,
            save_prediction=False,
            transfer_dir=None,
            transfer_extra="",
            mode="train",
            split_sample=0.0,
            split_by="y",
            split_train=False,
            split_test=False,
            train_y=0.7,
            node_mode=self.node_mode,
            early_stopping_patience=self.early_stopping_patience,
        )

    @staticmethod
    def _prepare_data(X: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        # Original GRAPE expects concrete edge values plus an explicit mask.
        # For matrix input adaptation we only fill NaNs with zero and keep the
        # missing pattern in `missing_mask`.
        missing_mask = np.isnan(X)
        X_filled = np.nan_to_num(X, nan=0.0)
        return pd.DataFrame(X_filled), missing_mask

    def _fit(self, X: np.ndarray) -> None:
        df_X, missing_mask = self._prepare_data(X)
        data = get_data_fix_mask(
            df_X,
            missing_mask,
            node_mode=self.node_mode,
            seed=0 if self.random_state is None else self.random_state,
        )
        args = self._build_args()
        filled_X, impute_model, model = train_gnn_mdi(
            data,
            args,
            device=self.device,
            return_filled_X=True,
        )
        self._model = model
        self._impute_model = impute_model
        self._fit_imputed_ = np.where(np.isnan(X), filled_X, X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._impute_model is None:
            raise RuntimeError("GRAPE model is not fitted.")
        df_X, missing_mask = self._prepare_data(X)
        data = get_data_fix_mask(
            df_X,
            missing_mask,
            node_mode=self.node_mode,
            seed=0 if self.random_state is None else self.random_state,
        )
        filled_X = out_of_sample_test_gnn_mdi(
            data,
            self._impute_model,
            self._model,
            self.device,
            return_filled_X=True,
        )
        return np.where(np.isnan(X), filled_X, X)

    def fit_transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        self.fit(X)
        if self._fit_imputed_ is None:
            raise RuntimeError("GRAPE fit_transform cache missing after fit.")
        return np.array(self._fit_imputed_, copy=True)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "model_types": self.model_types,
                "post_hiddens": self.post_hiddens,
                "concat_states": self.concat_states,
                "norm_embs": self.norm_embs,
                "aggr": self.aggr,
                "node_dim": self.node_dim,
                "edge_dim": self.edge_dim,
                "edge_mode": self.edge_mode,
                "gnn_activation": self.gnn_activation,
                "impute_hiddens": self.impute_hiddens,
                "impute_activation": self.impute_activation,
                "max_epochs": self.max_epochs,
                "opt": self.opt,
                "opt_scheduler": self.opt_scheduler,
                "opt_decay_step": self.opt_decay_step,
                "opt_decay_rate": self.opt_decay_rate,
                "dropout": self.dropout,
                "weight_decay": self.weight_decay,
                "learning_rate": self.learning_rate,
                "known": self.known,
                "loss_mode": self.loss_mode,
                "node_mode": self.node_mode,
                "early_stopping_patience": self.early_stopping_patience,
                "device": self.device,
                "direct_original_flow": True,
            }
        )
        return params
