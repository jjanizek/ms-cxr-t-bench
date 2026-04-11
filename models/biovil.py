"""BioViL (static) model wrapper.

BioViL uses a ResNet-50 image encoder trained with radiology report contrastive
learning (Bannur et al., ECCV 2022). Available via the hi-ml-multimodal package.

Install: pip install hi-ml-multimodal
"""
import logging
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from models.base import BaseModel

logger = logging.getLogger(__name__)


class BioViL(BaseModel):
    """Wrapper around Microsoft BioViL image encoder."""

    def __init__(self, device: str = "cpu"):
        try:
            from health_multimodal.image import get_biovil_resnet_inference
            self._model = get_biovil_resnet_inference()
        except ImportError as e:
            raise ImportError(
                "hi-ml-multimodal is required for BioViL. "
                "Install with: pip install hi-ml-multimodal"
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
        result = self._model.get_projected_global_embedding(img)
        return result  # (B, D)

    def encode_image_pair(self, img1: Tensor, img2: Tensor) -> Tensor:
        f1 = self.encode_image(img1)
        f2 = self.encode_image(img2)
        return torch.cat([f1, f2], dim=1)

    @torch.no_grad()
    def encode_text(self, text: list[str]) -> Optional[Tensor]:
        from health_multimodal.text import get_cxr_bert_inference
        text_model = get_cxr_bert_inference().to(self._device)
        return text_model.get_projected_embeddings(text)
