"""Matrix adapter for the vendored MissDiff diffusion core.

MissDiff is retained as an imputation candidate through its public source.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import numpy as np

from imputers.base import BaseImputer


try:
    import torch
    from torch.utils.data import DataLoader

    HAS_MISSDIFF_DEPS = find_spec("Missdiff_SDE") is not None
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None
    HAS_MISSDIFF_DEPS = False


def _load_core():
    from Missdiff_SDE.model import MLPDiffusion, Model
    from Missdiff_SDE.diffusion_utils import impute_mask

    return MLPDiffusion, Model, impute_mask


class MissDiffImputer(BaseImputer):
    """Official-loss MissDiff with bounded training and batched sampling."""

    name = "missdiff"

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        max_epochs: int = 20,
        batch_size: int = 1024,
        learning_rate: float = 1e-4,
        num_steps: int = 20,
        num_trials: int = 1,
        inference_batch_size: int = 2048,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        if not HAS_MISSDIFF_DEPS:
            raise ImportError("missdiff requires torch and the vendored official source.")
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.num_steps = max(2, int(num_steps))
        self.num_trials = max(1, int(num_trials))
        self.inference_batch_size = int(inference_batch_size)
        self.device = str(device)
        self._fit_imputed: np.ndarray | None = None

    def _run(self, X: np.ndarray) -> np.ndarray:
        if self.random_state is not None:
            np.random.seed(int(self.random_state))
            torch.manual_seed(int(self.random_state))
        MLPDiffusion, Model, impute_mask = _load_core()
        device = torch.device(
            self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        observed = ~np.isnan(X)
        missing = ~observed
        mean = np.nanmean(X, axis=0)
        scale = np.nanstd(X, axis=0)
        mean[~np.isfinite(mean)] = 0.0
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        standardized = (X - mean) / scale / 2.0
        initial = np.nan_to_num(standardized, nan=0.0).astype(np.float32)
        mask = missing.astype(np.float32)
        combined = np.concatenate([initial, mask], axis=1)
        loader = DataLoader(
            torch.as_tensor(combined),
            batch_size=min(self.batch_size, len(X)), shuffle=True,
            num_workers=0,
        )
        denoiser = MLPDiffusion(X.shape[1], self.hidden_dim).to(device)
        model = Model(denoise_fn=denoiser, hid_dim=X.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        model.train()
        for _ in range(self.max_epochs):
            for batch in loader:
                loss = model(batch.float().to(device)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        model.eval()
        trial_outputs: list[np.ndarray] = []
        for _ in range(self.num_trials):
            pieces: list[np.ndarray] = []
            for start in range(0, len(X), self.inference_batch_size):
                stop = min(start + self.inference_batch_size, len(X))
                x_chunk = torch.as_tensor(initial[start:stop], device=device)
                m_chunk = torch.as_tensor(mask[start:stop], device=device)
                sampled = impute_mask(
                    model.denoise_fn_D, x_chunk, m_chunk,
                    stop - start, X.shape[1], self.num_steps, str(device),
                    progress=False,
                )
                combined_chunk = sampled * m_chunk + x_chunk * (1.0 - m_chunk)
                pieces.append(combined_chunk.detach().cpu().numpy())
            trial_outputs.append(np.concatenate(pieces, axis=0))
        filled_standardized = np.mean(trial_outputs, axis=0)
        filled = filled_standardized * 2.0 * scale + mean
        filled[observed] = X[observed]
        return filled

    def _fit(self, X: np.ndarray) -> None:
        self._fit_imputed = self._run(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._fit_imputed is not None and self._fit_imputed.shape == X.shape:
            result, self._fit_imputed = self._fit_imputed, None
            return np.array(result, copy=True)
        return self._run(X)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update({
            "hidden_dim": self.hidden_dim,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_steps": self.num_steps,
            "num_trials": self.num_trials,
            "inference_batch_size": self.inference_batch_size,
            "device": self.device,
            "source_package": "Missdiff_SDE",
        })
        return params
