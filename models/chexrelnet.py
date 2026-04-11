"""CheXRelNet model wrapper.

CheXRelNet (Dalla Serra et al., 2022) is a graph-based model for encoding
relational information between chest X-ray image pairs.

This wrapper expects the model checkpoint and codebase to be available.
"""
import logging
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from models.base import BaseModel

logger = logging.getLogger(__name__)


class CheXRelNet(BaseModel):
    """Wrapper for CheXRelNet relational chest X-ray encoder."""

    def __init__(self, checkpoint: Optional[str] = None, device: str = "cpu"):
        self._device = torch.device(device)
        # TODO: implement once CheXRelNet checkpoint source is confirmed
        raise NotImplementedError(
            "CheXRelNet wrapper is not yet implemented. "
            "Provide checkpoint path and implement model loading."
        )

    def to(self, device):
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self

    def eval(self):
        self._model.eval()

    @torch.no_grad()
    def encode_image(self, img: Tensor) -> Tensor:
        raise NotImplementedError

    def encode_image_pair(self, img1: Tensor, img2: Tensor) -> Tensor:
        raise NotImplementedError
