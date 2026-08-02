"""HIVAE imputer adapted for the local research framework."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from imputers.base import BaseImputer
from utils.logging_utils import get_logger, log_kv

try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    import external_dependencies.HIVAE.graph_new as graph_new
    import external_dependencies.HIVAE.read_functions as read_functions

    HAS_TENSORFLOW = True
except ImportError:  # pragma: no cover
    tf = None
    graph_new = None
    read_functions = None
    HAS_TENSORFLOW = False


class HIVAEImputer(BaseImputer):
    """Numeric-only HIVAE wrapper.

    The original HI-VAE method targets heterogeneous data. In this project
    phase we only support numeric columns, so this wrapper generates a temporary
    types file with all variables marked as `real` and delegates the core graph
    construction to the vendored HI-VAE implementation.
    """

    name = "hivae"

    def __init__(
        self,
        *,
        dim_latent_z: int = 2,
        dim_latent_y: int = 3,
        dim_latent_s: int = 4,
        batch_size: int = 128,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        model_name: str = "model_HIVAE_inputDropout",
        display_epoch: int = 10,
        **kwargs: Any,
    ) -> None:
        if not HAS_TENSORFLOW:
            raise ImportError("hivae requires the optional dependency 'tensorflow'.")
        super().__init__(**kwargs)
        self.dim_latent_z = dim_latent_z
        self.dim_latent_y = dim_latent_y
        self.dim_latent_s = dim_latent_s
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.display_epoch = display_epoch
        self.logger = get_logger("imputation_reco.hivae")
        self.trained = False
        self.graph = None
        self.tf_nodes: dict[str, Any] = {}
        self._types_dict: list[dict[str, Any]] | None = None
        self._types_file: Path | None = None
        runtime_root = Path(__file__).resolve().parents[1] / "results" / "hivae_runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = runtime_root / f"run_{uuid4().hex[:8]}"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = str(self._checkpoint_dir / "model.ckpt")

    def _build_numeric_types_dict(self, n_features: int) -> list[dict[str, Any]]:
        return [{"type": "real", "dim": 1} for _ in range(n_features)]

    def _write_types_file(self, types_dict: list[dict[str, Any]]) -> Path:
        path = self._checkpoint_dir / "types.csv"
        with path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=["type", "dim"])
            writer.writeheader()
            for row in types_dict:
                writer.writerow({"type": row["type"], "dim": int(row["dim"])})
        return path

    def _encode_data(self, data: np.ndarray, types_dict: list[dict[str, Any]]) -> np.ndarray:
        data_masked = np.ma.masked_where(np.isnan(data), data)
        data_filler = [0.0 for _ in range(len(types_dict))]
        data = data_masked.filled(data_filler)
        data_complete = []
        for col_idx in range(data.shape[1]):
            data_complete.append(np.transpose([data[:, col_idx]]))
        return np.concatenate(data_complete, axis=1)

    def _split_data_by_variable(self, X_enc: np.ndarray, types_dict: list[dict[str, Any]]) -> list[np.ndarray]:
        data_list = []
        idx_start = 0
        for col_info in types_dict:
            var_dim = int(col_info["dim"])
            data_list.append(X_enc[:, idx_start : idx_start + var_dim])
            idx_start += var_dim
        return data_list

    def _batch_iterator(self, X_enc: np.ndarray, mask: np.ndarray, types_dict: list[dict[str, Any]]):
        n_rows = X_enc.shape[0]
        n_batches = math.ceil(n_rows / self.batch_size)
        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, n_rows)
            batch_X_enc = X_enc[start:end]
            batch_mask = mask[start:end]
            if end - start < self.batch_size:
                remaining = self.batch_size - (end - start)
                repeat_indices = np.arange(remaining) % n_rows
                batch_X_enc = np.concatenate((batch_X_enc, X_enc[repeat_indices]), axis=0)
                batch_mask = np.concatenate((batch_mask, mask[repeat_indices]), axis=0)
            batch_data_list = self._split_data_by_variable(batch_X_enc, types_dict)
            batch_data_list_observed = [
                batch_data_list[i] * np.reshape(batch_mask[:, i], [self.batch_size, 1])
                for i in range(len(batch_data_list))
            ]
            yield batch_data_list, batch_mask.astype(int), batch_data_list_observed

    def _build_model(self, types_file: Path) -> None:
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_nodes = graph_new.HVAE_graph(
                model_name=self.model_name,
                types_file=str(types_file),
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                z_dim=self.dim_latent_z,
                y_dim=self.dim_latent_y,
                s_dim=self.dim_latent_s,
                y_dim_partition=[],
            )

    def _fit(self, X: np.ndarray) -> None:
        types_dict = self._build_numeric_types_dict(X.shape[1])
        self._types_dict = types_dict
        self._types_file = self._write_types_file(types_dict)
        self._build_model(self._types_file)

        X_enc = self._encode_data(X, types_dict)
        mask = (~np.isnan(X)).astype(int)
        n_rows = X_enc.shape[0]
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            for epoch in range(self.epochs):
                indices = np.random.permutation(n_rows)
                X_enc_epoch = X_enc[indices]
                mask_epoch = mask[indices]
                tau = max(1.0 - 0.01 * epoch, 1e-3)
                tau2 = min(0.001 * epoch, 1.0)
                avg_loss = 0.0
                for batch_data_list, batch_mask, batch_data_list_observed in self._batch_iterator(X_enc_epoch, mask_epoch, types_dict):
                    feed_dict = {placeholder: value for placeholder, value in zip(self.tf_nodes["ground_batch"], batch_data_list)}
                    feed_dict.update({placeholder: value for placeholder, value in zip(self.tf_nodes["ground_batch_observed"], batch_data_list_observed)})
                    feed_dict[self.tf_nodes["miss_list"]] = batch_mask
                    feed_dict[self.tf_nodes["tau_GS"]] = tau
                    feed_dict[self.tf_nodes["tau_var"]] = tau2
                    _, loss_re = sess.run([self.tf_nodes["optim"], self.tf_nodes["loss_re"]], feed_dict=feed_dict)
                    avg_loss += float(np.mean(loss_re))
                if (epoch + 1) % self.display_epoch == 0 or epoch == 0:
                    log_kv(self.logger, "hivae training", epoch=epoch + 1, avg_loss=round(avg_loss, 6))
            saver.save(sess, self._checkpoint_path)
        self.trained = True

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not self.trained or self.graph is None or self._types_dict is None:
            raise RuntimeError("HIVAE model is not fitted.")
        X_enc = self._encode_data(X, self._types_dict)
        mask = (~np.isnan(X)).astype(int)
        p_params_list = []
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self._checkpoint_path)
            tau = 1e-3
            for batch_data_list, batch_mask, batch_data_list_observed in self._batch_iterator(X_enc, mask, self._types_dict):
                feed_dict = {placeholder: value for placeholder, value in zip(self.tf_nodes["ground_batch"], batch_data_list)}
                feed_dict.update({placeholder: value for placeholder, value in zip(self.tf_nodes["ground_batch_observed"], batch_data_list_observed)})
                feed_dict[self.tf_nodes["miss_list"]] = batch_mask
                feed_dict[self.tf_nodes["tau_GS"]] = tau
                feed_dict[self.tf_nodes["tau_var"]] = 1.0
                _, _, _, test_params = sess.run(
                    [
                        self.tf_nodes["samples_test"],
                        self.tf_nodes["log_p_x_test"],
                        self.tf_nodes["log_p_x_missing_test"],
                        self.tf_nodes["test_params"],
                    ],
                    feed_dict=feed_dict,
                )
                p_params_list.append(test_params)
        p_params_complete = read_functions.p_distribution_params_concatenation(
            p_params_list,
            self._types_dict,
            self.dim_latent_z,
            self.dim_latent_s,
        )
        _, loglik_mode = read_functions.statistics(p_params_complete["x"], self._types_dict, df_size=X_enc.shape[0])
        X_imputed = np.where(mask.astype(bool), X, np.round(loglik_mode, 3))
        X_imputed = np.asarray(X_imputed, dtype=float)
        if not np.isfinite(X_imputed).all():
            raise RuntimeError("HIVAE produced NaN or infinite values; source-only mode forbids fallback imputation.")
        return X_imputed

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "dim_latent_z": self.dim_latent_z,
                "dim_latent_y": self.dim_latent_y,
                "dim_latent_s": self.dim_latent_s,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "model_name": self.model_name,
                "display_epoch": self.display_epoch,
                "checkpoint_path": self._checkpoint_path,
            }
        )
        return params
