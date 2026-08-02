"""Official ReMasker adapter using the CACTI authors' maintained baseline."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

from imputers.base import BaseImputer


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "external_dependencies" / "cacti_official"

try:  # Keep imports lazy: registry discovery should not initialize torch/timm.
    import torch

    HAS_REMASKER_DEPS = OFFICIAL_ROOT.exists()
except Exception:  # pragma: no cover - optional dependency
    torch = None
    HAS_REMASKER_DEPS = False

def _load_official() -> tuple[Any, Any]:
    if torch is None:
        raise ImportError("ReMasker requires torch.")
    # Import the installed torchvision extension before timm.  Older CPU-only
    # wheels needed an NMS schema shim, but defining that shim against the
    # current compatible wheel causes a fatal duplicate registration.
    import torchvision  # noqa: F401
    if str(OFFICIAL_ROOT) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_ROOT))
    from src.imputers.randmae import RandomMAE
    from src.loaders.baseloader import BaseDataset

    return RandomMAE, BaseDataset


class ReMaskerImputer(BaseImputer):
    """ReMasker (ICLR 2024), with equivalent batched CPU inference."""

    name = "remasker"
    cost = 0.85

    def __init__(
        self,
        *,
        max_epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        mask_ratio: float = 0.5,
        embed_dim: int = 32,
        encoder_depth: int = 4,
        decoder_depth: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.mask_ratio = float(mask_ratio)
        self.embed_dim = int(embed_dim)
        self.encoder_depth = int(encoder_depth)
        self.decoder_depth = int(decoder_depth)
        self._official: Any | None = None
        self._base_dataset: Any | None = None
        self._min_scale: np.ndarray | None = None
        self._max_scale: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        if not HAS_REMASKER_DEPS:
            raise ImportError("Official CACTI/ReMasker source or torch is unavailable.")
        RandomMAE, BaseDataset = _load_official()
        seed = 0 if self.random_state is None else int(self.random_state)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._min_scale = np.nanmin(X, axis=0).astype(np.float32)
        self._max_scale = np.nanmax(X, axis=0).astype(np.float32)
        args = SimpleNamespace(
            batch_size=self.batch_size,
            min_lr=5e-6,
            lr=self.learning_rate,
            grad_clip=5.0,
            warmup_epochs=max(1, min(5, self.max_epochs // 4)),
            epochs=self.max_epochs,
            mask_ratio=self.mask_ratio,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_workers=0,
            weight_decay=1e-3,
            embed_dim=self.embed_dim,
            nencoder=self.encoder_depth,
            ndecoder=self.decoder_depth,
            checkpoint_path=None,
        )
        self._official = RandomMAE(args, feats=list(range(X.shape[1])))
        self._official.min_scale = self._min_scale
        self._official.max_scale = self._max_scale
        self._base_dataset = BaseDataset
        self._official.fit(torch.tensor(X, dtype=torch.float32))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._official is None or self._base_dataset is None:
            raise RuntimeError("ReMasker is not fitted.")
        original = torch.tensor(X, dtype=torch.float32)
        min_scale = torch.tensor(self._min_scale, dtype=torch.float32)
        max_scale = torch.tensor(self._max_scale, dtype=torch.float32)
        scaled = (original - min_scale) / (max_scale - min_scale + 1e-6)
        loader = torch.utils.data.DataLoader(
            self._base_dataset(scaled), batch_size=self.batch_size,
            shuffle=False, num_workers=0,
        )
        device = self._official.device
        self._official.model.to(device).eval()
        predictions = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                _, estimate = self._official.model(batch["table"], batch["miss_map"])
                predictions.append(estimate.cpu())
        imputed = torch.cat(predictions, dim=0)
        imputed = imputed * (max_scale - min_scale + 1e-6) + min_scale
        result = torch.where(torch.isnan(original), imputed, original)
        return result.numpy()

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update({
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "mask_ratio": self.mask_ratio,
            "embed_dim": self.embed_dim,
            "encoder_depth": self.encoder_depth,
            "decoder_depth": self.decoder_depth,
            "official_source": "sriramlab/CACTI maintained ReMasker baseline",
            "batched_inference": True,
        })
        return params
