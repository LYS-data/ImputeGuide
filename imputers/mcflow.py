"""Official-source MCFlow (CVPR 2020) matrix adapter."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

from imputers.base import BaseImputer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "DiffPuter-main" / "baselines" / "MCFlow"

try:
    import torch

    HAS_MCFLOW_DEPS = SOURCE.is_dir()
except Exception:  # pragma: no cover
    torch = None
    HAS_MCFLOW_DEPS = False


def _load_core():
    baselines = SOURCE.parent
    if str(baselines) not in sys.path:
        sys.path.insert(0, str(baselines))
    from MCFlow.loader import DataLoader as MCFlowData
    from MCFlow.models import InterpRealNVP, LatentToLatentApprox
    from MCFlow import util

    return MCFlowData, InterpRealNVP, LatentToLatentApprox, util


class MCFlowImputer(BaseImputer):
    """Transductive MCFlow with the official training and reset schedule."""

    name = "mcflow"

    def __init__(
        self,
        *,
        n_epochs: int = 20,
        batch_size: int = 256,
        num_nf_layers: int = 3,
        learning_rate: float = 1e-4,
        reset_imputations: bool = True,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        if not HAS_MCFLOW_DEPS:
            raise ImportError("mcflow requires torch and the vendored official source.")
        super().__init__(**kwargs)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.num_nf_layers = int(num_nf_layers)
        self.learning_rate = float(learning_rate)
        self.reset_imputations = bool(reset_imputations)
        self.device = str(device)
        self._fit_imputed: np.ndarray | None = None

    def _run(self, X: np.ndarray) -> np.ndarray:
        if self.random_state is not None:
            np.random.seed(int(self.random_state))
            torch.manual_seed(int(self.random_state))
        MCFlowData, InterpRealNVP, LatentToLatentApprox, util = _load_core()
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        mask = np.isnan(X).astype(np.float32)
        observed = ~np.isnan(X)
        # The official loader requires a complete carrier matrix, but its loss
        # masks missing entries. Observed-column means are therefore a neutral
        # placeholder, not ground truth for missing cells.
        means = np.nanmean(X, axis=0)
        means[~np.isfinite(means)] = 0.0
        carrier = np.where(observed, X, means).astype(np.float32)
        dataset = MCFlowData(
            mode=0, seed=int(self.random_state or 0), path="matrix",
            train_X=carrier, test_X=carrier,
            train_mask=mask, test_mask=mask,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=min(self.batch_size, len(X)),
            shuffle=True, drop_last=False, num_workers=0,
        )
        eval_loader = torch.utils.data.DataLoader(
            dataset, batch_size=min(self.batch_size, len(X)),
            shuffle=False, drop_last=False, num_workers=0,
        )
        d = X.shape[1]
        args = SimpleNamespace(
            use_cuda=use_cuda, n_epochs=self.n_epochs,
            num_nf_layers=self.num_nf_layers, lr=self.learning_rate,
            dataset="matrix", disable_progress=True,
        )
        flow = util.init_flow_model(d, self.num_nf_layers, InterpRealNVP, d, args)
        hidden = [d, d, d, d, d]
        latent = LatentToLatentApprox(d, hidden).float()
        if use_cuda:
            latent = latent.cuda()
        flow_optimizer = torch.optim.Adam(
            [parameter for parameter in flow.parameters() if parameter.requires_grad],
            lr=self.learning_rate,
        )
        latent_optimizer = torch.optim.Adam(
            [parameter for parameter in latent.parameters() if parameter.requires_grad],
            lr=self.learning_rate,
        )
        reset_at = 2
        for epoch in range(self.n_epochs):
            util.endtoend_train(
                flow, latent, flow_optimizer, latent_optimizer, loader, args, epoch,
            )
            if self.reset_imputations and (epoch + 1) == reset_at:
                dataset.reset_imputed_values(latent, flow, int(self.random_state or 0), args)
                flow = util.init_flow_model(d, self.num_nf_layers, InterpRealNVP, d, args)
                flow_optimizer = torch.optim.Adam(
                    [parameter for parameter in flow.parameters() if parameter.requires_grad],
                    lr=self.learning_rate,
                )
                reset_at *= 2
        dataset.mode = 0
        filled_scaled, _ = util.get_filled_data(flow, latent, eval_loader, args)
        filled = np.asarray(
            dataset.min_max_scaler_train.inverse_transform(filled_scaled),
            dtype=float,
        )
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
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "num_nf_layers": self.num_nf_layers,
            "learning_rate": self.learning_rate,
            "reset_imputations": self.reset_imputations,
            "device": self.device,
            "source_path": str(SOURCE),
            "source": "official CVPR 2020 MCFlow training flow",
            "compatibility_fix": (
                "both official losses are backpropagated before optimizer steps "
                "to satisfy modern PyTorch graph versioning; the official "
                "inference remainder starts from iterations*256 so tables "
                "smaller than one batch are supported"
            ),
        })
        return params
