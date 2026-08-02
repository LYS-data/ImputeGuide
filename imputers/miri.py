"""Leakage-free adapter for the official NeurIPS 2025 MIRI core."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from imputers.base import BaseImputer


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "external_dependencies" / "MIRI-Imputation"

try:
    import torch

    HAS_MIRI_DEPS = OFFICIAL_ROOT.exists()
except Exception:  # pragma: no cover - optional dependency
    torch = None
    HAS_MIRI_DEPS = False


def _load_official_mlp():
    path = OFFICIAL_ROOT / "src" / "mlp.py"
    spec = importlib.util.spec_from_file_location("_miri_official_mlp", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official MIRI MLP from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MLP


def _sample_ode(model, state, steps: int):
    d = state.shape[1] // 3
    dt = 1.0 / max(int(steps), 1)
    with torch.no_grad():
        for step in range(max(int(steps), 1)):
            time = torch.full(
                (state.shape[0], 1), step * dt,
                dtype=state.dtype, device=state.device,
            )
            velocity = model(state, time)
            state[:, :d] += velocity * dt
            state[:, d:2 * d] += velocity * dt
    return state[:, :d]


class MIRIImputer(BaseImputer):
    """Mutual-information-reducing rectified-flow imputation.

    The official demo accepts ``Xstar`` only to print MMD/MI diagnostics during
    optimization.  This adapter omits those diagnostics so neither complete
    values nor downstream labels enter fitting; the vector-field objective and
    iterative ODE update are unchanged.
    """

    name = "miri"
    cost = 0.95

    def __init__(
        self,
        *,
        max_rounds: int = 3,
        max_epochs: int = 10,
        ode_steps: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        inference_batch_size: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_rounds = int(max_rounds)
        self.max_epochs = int(max_epochs)
        self.ode_steps = int(ode_steps)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.inference_batch_size = int(inference_batch_size)
        self._filled: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        if not HAS_MIRI_DEPS or torch is None:
            raise ImportError("MIRI requires PyTorch and the official source tree.")
        seed = 0 if self.random_state is None else int(self.random_state)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MLP = _load_official_mlp()

        observed = ~np.isnan(X)
        center = np.nanmean(X, axis=0)
        scale = np.nanstd(X, axis=0)
        center[~np.isfinite(center)] = 0.0
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        standardized = (X - center) / scale
        rng = np.random.default_rng(seed)
        initial = np.where(
            observed, standardized, rng.standard_normal(X.shape),
        ).astype(np.float32)
        values = torch.tensor(initial, dtype=torch.float32)
        mask = torch.tensor(observed.astype(np.float32))
        d = X.shape[1]
        criterion = torch.nn.MSELoss()

        shuffled = values[torch.randperm(len(values))].clone()
        for _ in range(max(self.max_rounds, 1)):
            model = MLP(d).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.learning_rate,
            )
            dataset = torch.utils.data.TensorDataset(
                torch.cat([values, mask], dim=1), shuffled,
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True,
            )
            model.train()
            for _ in range(max(self.max_epochs, 1)):
                for packed, target in loader:
                    source = packed[:, :d].to(device)
                    observed_mask = packed[:, d:].to(device)
                    target = target.to(device)
                    time = torch.rand(len(target), 1, device=device)
                    interpolated = time * target + (1.0 - time) * source
                    left = interpolated.clone()
                    right = interpolated.clone()
                    left[observed_mask == 1] = source[observed_mask == 1]
                    right[observed_mask == 1] = target[observed_mask == 1]
                    state = torch.cat([left, right, observed_mask], dim=1)
                    velocity = model(state, time)
                    missing_cells = observed_mask == 0
                    if not bool(missing_cells.any()):
                        continue
                    loss = criterion(
                        velocity[missing_cells],
                        (target - source)[missing_cells],
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

            model.eval()
            state = torch.cat([values, values, mask], dim=1)
            for start in range(0, len(values), self.inference_batch_size):
                stop = min(start + self.inference_batch_size, len(values))
                estimate = _sample_ode(
                    model, state[start:stop].clone().to(device), self.ode_steps,
                ).cpu()
                missing_cells = mask[start:stop] == 0
                chunk = values[start:stop].clone()
                chunk[missing_cells] = estimate[missing_cells]
                values[start:stop] = chunk
            shuffled = values[torch.randperm(len(values))].clone()

        result = values.numpy().astype(float) * scale + center
        result[observed] = X[observed]
        self._filled = result

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._filled is None or self._filled.shape != X.shape:
            raise RuntimeError("MIRI is transductive and must transform its fitted matrix.")
        result = self._filled.copy()
        observed = ~np.isnan(X)
        result[observed] = X[observed]
        return result

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update({
            "max_rounds": self.max_rounds,
            "max_epochs": self.max_epochs,
            "ode_steps": self.ode_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "inference_batch_size": self.inference_batch_size,
            "official_source": "yujhml/MIRI-Imputation",
            "ground_truth_diagnostics_disabled": True,
        })
        return params
