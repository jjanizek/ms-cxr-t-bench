"""BioViL-T (temporal) model wrapper.

BioViL-T uses a MultiImageEncoder: a shared ResNet50 backbone + ViT temporal
pooler that encodes a (prior, current) image pair jointly into a single 128-dim
projected embedding.  This is fundamentally different from simple concatenation
— the temporal difference is explicitly modelled.

The model is downloaded automatically from HuggingFace via hi-ml-multimodal.

Install: pip install hi-ml-multimodal

Reference: Bannur et al., "Learning to Exploit Temporal Structure for
Biomedical Vision-Language Processing", CVPR 2023.
"""
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from models.base import BaseModel

logger = logging.getLogger(__name__)


class BioViLT(BaseModel):
    """Wrapper around Microsoft BioViL-T temporal image encoder.

    encode_image_pair(img1, img2) uses the full temporal forward:
        MultiImageEncoder(current=img2, previous=img1) → 128-dim embedding.

    encode_image(img) falls back to the single-image path (previous=None),
    which uses a learned "missing prior" placeholder — useful for ablations
    but not the intended temporal use case.
    """

    # BioViL-T preprocessing: resize to 512, center crop to 448
    RESIZE = 512
    CROP = 448

    def __init__(self, device: str = "cpu"):
        try:
            from health_multimodal.image.utils import get_image_inference, ImageModelType
            from health_multimodal.image.data.transforms import (
                create_chest_xray_transform_for_inference,
            )
        except ImportError as e:
            raise ImportError(
                "hi-ml-multimodal is required for BioViL-T.\n"
                "Install with: pip install hi-ml-multimodal"
            ) from e

        self._device = torch.device(device)
        engine = get_image_inference(ImageModelType.BIOVIL_T)
        self._model = engine.model.to(self._device).eval()
        self._transform = create_chest_xray_transform_for_inference(
            resize=self.RESIZE, center_crop_size=self.CROP
        )
        logger.info("BioViL-T loaded.")

    def to(self, device):
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self

    def eval(self):
        self._model.eval()

    def _apply_transform(self, img: Tensor) -> Tensor:
        """Apply BioViL-T's own preprocessing to a batch of (C,H,W) float tensors.

        Expects img in [0,1] range.  The hi-ml transform operates on PIL Images;
        we convert back via torchvision.
        """
        from torchvision.transforms.functional import to_pil_image
        processed = []
        for i in range(img.shape[0]):
            pil = to_pil_image(img[i].cpu())
            processed.append(self._transform(pil))
        return torch.stack(processed).to(self._device)

    @torch.no_grad()
    def encode_image(self, img: Tensor) -> Tensor:
        """Single-image encoding (no temporal context). Returns (B, 128)."""
        img = self._apply_transform(img)
        out = self._model(img)
        emb = F.normalize(out.projected_global_embedding, dim=-1)
        return emb

    @torch.no_grad()
    def encode_image_pair(self, img1: Tensor, img2: Tensor) -> Tensor:
        """Temporal pair encoding. img1=prior, img2=current. Returns (B, 128).

        Routes through MultiImageEncoder.forward(current=img2, previous=img1)
        which computes a joint temporal representation capturing the difference
        between the two studies.
        """
        img1 = self._apply_transform(img1)
        img2 = self._apply_transform(img2)

        # Go through encoder with both images, then project
        encoder = self._model.encoder
        patch_fused, avg_pooled = encoder(
            current_image=img2,
            previous_image=img1,
            return_patch_embeddings=True,
        )
        out = self._model.forward_post_encoder(patch_fused, avg_pooled)
        emb = F.normalize(out.projected_global_embedding, dim=-1)
        return emb  # (B, 128)
