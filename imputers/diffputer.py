"""DiffPuter wrapper that reuses the official core model and diffusion sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from imputers.base import BaseImputer

try:
    import torch
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader

    from external_dependencies.diffputer_original import MLPDiffusion, Model, impute_mask

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    ReduceLROnPlateau = None
    DataLoader = None
    MLPDiffusion = None
    Model = None
    impute_mask = None
    HAS_TORCH = False


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


def _original_mean_std(data: np.ndarray, mask: np.ndarray) -> NormalizationStats:
    """Match the official DiffPuter mean/std computation.

    The original implementation expects ``mask`` to be True on missing entries.
    """

    observed_mask = (~mask).astype(np.float32)
    observed_count = observed_mask.sum(0)
    observed_count[observed_count == 0] = 1
    safe_data = np.where(mask, 0.0, data)
    mean = safe_data.sum(0) / observed_count
    var = (((data - mean) ** 2) * observed_mask).sum(0) / observed_count
    std = np.sqrt(var)
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where((~np.isfinite(std)) | (std < 1e-6), 1.0, std)
    return NormalizationStats(mean=mean.astype(float), std=std.astype(float))


def _normalize(X: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (X - stats.mean) / stats.std / 2.0


def _denormalize(X: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return X * 2.0 * stats.std + stats.mean


class DiffPuterImputer(BaseImputer):
    """Official-core DiffPuter adapted to the project's matrix imputer interface.

    This wrapper keeps the original MLP diffusion model, EDM loss, training
    loop structure, scheduler, and masked sampling procedure. The only major
    adaptation is replacing the repository's dataset-specific CLI pipeline with
    direct ``numpy.ndarray`` input and output.
    """

    name = "diffputer"

    def __init__(
        self,
        *,
        hid_dim: int = 1024,
        max_iter: int = 1,
        num_trials: int = 20,
        num_steps: int = 50,
        max_epochs: int = 10001,
        batch_size: int = 4096,
        learning_rate: float = 1e-4,
        early_stopping_patience: int = 500,
        device: str | None = None,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("diffputer requires the optional dependency 'torch'.")
        super().__init__(**kwargs)
        self.hid_dim = hid_dim
        self.max_iter = max_iter
        self.num_trials = num_trials
        self.num_steps = num_steps
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers

        self._stats: NormalizationStats | None = None
        self._model_state: dict[str, torch.Tensor] | None = None
        self._fit_imputed_: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        missing_mask = np.isnan(X)
        self._stats = _original_mean_std(X, missing_mask)
        x_norm = _normalize(X, self._stats)
        x_tensor = torch.tensor(np.nan_to_num(x_norm, nan=0.0), dtype=torch.float32)
        mask_tensor = torch.tensor(missing_mask, dtype=torch.float32)

        current_train_data = ((1.0 - mask_tensor) * x_tensor).cpu().numpy()
        current_model_state: dict[str, torch.Tensor] | None = None
        final_rec = x_tensor.clone()

        for _iteration in range(self.max_iter):
            model, best_state = self._train_single_iteration(current_train_data, in_dim=X.shape[1])
            current_model_state = best_state
            model.load_state_dict(best_state)
            final_rec = self._in_sample_impute(model, x_tensor, mask_tensor)
            current_train_data = final_rec.detach().cpu().numpy()

        if current_model_state is None:
            raise RuntimeError("DiffPuter failed to produce a trained model state.")

        self._model_state = current_model_state
        imputed = _denormalize(final_rec.detach().cpu().numpy(), self._stats)
        self._fit_imputed_ = np.where(missing_mask, imputed, X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._stats is None or self._model_state is None:
            raise RuntimeError("DiffPuter must be fitted before transform().")
        missing_mask = np.isnan(X)
        x_norm = _normalize(X, self._stats)
        x_tensor = torch.tensor(np.nan_to_num(x_norm, nan=0.0), dtype=torch.float32)
        mask_tensor = torch.tensor(missing_mask, dtype=torch.float32)
        model = self._instantiate_model(in_dim=X.shape[1])
        model.load_state_dict(self._model_state)
        model.eval()
        rec = self._out_of_sample_impute(model, x_tensor, mask_tensor)
        imputed = _denormalize(rec.detach().cpu().numpy(), self._stats)
        return np.where(missing_mask, imputed, X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        if self._fit_imputed_ is None:
            raise RuntimeError("DiffPuter fit_transform cache missing after fit.")
        return np.array(self._fit_imputed_, copy=True)

    def _instantiate_model(self, *, in_dim: int):
        denoise_fn = MLPDiffusion(in_dim, self.hid_dim).to(self.device)
        return Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(self.device)

    def _train_single_iteration(self, train_data: np.ndarray, *, in_dim: int):
        batch_size = min(self.batch_size, max(len(train_data), 1))
        train_loader = DataLoader(
            train_data.astype(np.float32, copy=False),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        model = self._instantiate_model(in_dim=in_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=0.0)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=50)

        best_loss = float("inf")
        patience = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.train()
        for _epoch in range(self.max_epochs):
            batch_loss = 0.0
            len_input = 0
            for batch in train_loader:
                inputs = batch.float().to(self.device)
                loss = model(inputs)
                batch_loss += float(loss.item()) * len(inputs)
                len_input += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            curr_loss = batch_loss / max(len_input, 1)
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    break

        return model, best_state

    def _in_sample_impute(self, model, x_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
        recs = []
        model.eval()
        for _ in range(self.num_trials):
            x_miss = ((1.0 - mask_tensor.float()) * x_tensor).to(self.device)
            rec_x = impute_mask(
                model.denoise_fn_D,
                x_miss,
                mask_tensor,
                x_tensor.shape[0],
                x_tensor.shape[1],
                self.num_steps,
                self.device,
            )
            mask_int = mask_tensor.to(torch.float).to(self.device)
            rec_x = rec_x * mask_int + x_miss * (1 - mask_int)
            recs.append(rec_x)

        return torch.stack(recs, dim=0).mean(0)

    def _out_of_sample_impute(self, model, x_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
        recs = []
        model.eval()
        for _ in range(self.num_trials):
            x_miss = ((1.0 - mask_tensor.float()) * x_tensor).to(self.device)
            rec_x = impute_mask(
                model.denoise_fn_D,
                x_miss,
                mask_tensor,
                x_tensor.shape[0],
                x_tensor.shape[1],
                self.num_steps,
                self.device,
            )
            mask_int = mask_tensor.to(torch.float).to(self.device)
            rec_x = rec_x * mask_int + x_miss * (1 - mask_int)
            recs.append(rec_x)

        return torch.stack(recs, dim=0).mean(0)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "hid_dim": self.hid_dim,
                "max_iter": self.max_iter,
                "num_trials": self.num_trials,
                "num_steps": self.num_steps,
                "max_epochs": self.max_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "early_stopping_patience": self.early_stopping_patience,
                "device": self.device,
                "num_workers": self.num_workers,
            }
        )
        return params
