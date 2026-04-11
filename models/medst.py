"""Med-ST model wrapper.

Med-ST (SVT-Yang et al.) is cloned into external/MedST/.
Clone: git clone https://github.com/SVT-Yang/MedST external/MedST

This wrapper imports from the cloned repo and exposes the BaseModel interface.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from models.base import BaseModel

logger = logging.getLogger(__name__)

MEDST_DIR = Path(__file__).parent.parent / "external" / "MedST"


class MedST(BaseModel):
    """Wrapper around Med-ST temporal chest X-ray model."""

    def __init__(self, checkpoint: Optional[str] = None, device: str = "cpu"):
        if not MEDST_DIR.exists():
            raise RuntimeError(
                f"Med-ST repo not found at {MEDST_DIR}. "
                "Clone with: git clone https://github.com/SVT-Yang/MedST external/MedST"
            )
        sys.path.insert(0, str(MEDST_DIR))

        # Import is deferred to allow the module to load without the external repo
        try:
            # Adjust import path based on actual MedST repo structure
            from models import build_model as medst_build  # type: ignore
            self._model = medst_build(checkpoint=checkpoint)
        except ImportError as e:
            raise ImportError(
                f"Failed to import Med-ST from {MEDST_DIR}. "
                "Check that the repo is correctly cloned and dependencies are installed."
            ) from e

        self._device = torch.device(device)
        self._model = self._model.to(self._device).eval()

    def to(self, device):
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self

    def eval(self):
        self._model.eval()

    @torch.no_grad()
    def encode_image(self, img: Tensor) -> Tensor:
        img = img.to(self._device)
        # TODO: update method name once MedST API is confirmed
        return self._model.encode_image(img)

    def encode_image_pair(self, img1: Tensor, img2: Tensor) -> Tensor:
        f1 = self.encode_image(img1)
        f2 = self.encode_image(img2)
        return torch.cat([f1, f2], dim=1)
