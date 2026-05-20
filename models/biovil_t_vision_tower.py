"""BioViL-T encoder, wrapped to drop into MAIRA-2's vision_tower slot.

MAIRA-2 uses a Dinov2Backbone as its `vision_tower`. Downstream code (the
multi_modal_projector and LM's image-token plumbing) consumes
    out.feature_maps[0]  # shape (B, 1370, 768)
i.e. 37×37 patch tokens + 1 prefix token at hidden dim 768.

BioViL-T's MultiImageEncoder, when called single-image, returns a
(B, 512, 14, 14) patch grid (its avg-pooled 512-d global is discarded here).
This module bridges those two shapes:

  bv-patches (B,512,14,14)  →  bilinear upsample to (B,512,37,37)
                              →  flatten to (B,1369,512)
                              →  prepend a learned 512-d "prefix" token  →  (B,1370,512)
                              →  Linear(512, 768)                          →  (B,1370,768)
                              →  wrap in BackboneOutput(feature_maps=(x,))

The BioViL-T weights are frozen by default (we just want to test if its
features help); the adapter Linear + prefix token are the new trainable
parameters and are intended to be in the optimizer (modules_to_save in PEFT
or registered as separate trainables when we set up the optimizer).

Image-preprocessing path: MAIRA-2's processor preprocesses for rad-DINO
(518×518, rad-DINO normalisation). BioViL-T wants ImageNet mean/std at
448×448. We undo MAIRA-2's normalisation inside the wrapper and re-normalise
for BioViL-T, so the rest of the data pipeline is untouched.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# rad-DINO normalisation used by MAIRA-2's processor. Match what the processor
# applies so we can invert it. These are the standard MAIRA-2 / rad-DINO stats.
RADDINO_MEAN = (0.5307, 0.5307, 0.5307)
RADDINO_STD = (0.2583, 0.2583, 0.2583)

# BioViL-T uses ImageNet stats internally (its create_chest_xray_transform_for_inference
# applies ImageNet normalisation after converting grayscale to 3-channel).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Target spatial grid for the output: 37×37 patches + 1 prefix = 1370 tokens.
TARGET_GRID = 37
TARGET_TOKENS = TARGET_GRID * TARGET_GRID + 1  # 1370


@dataclass
class _BackboneOutput:
    """Mimic transformers BackboneOutput (we only need .feature_maps)."""
    feature_maps: tuple


@dataclass
class _VisionConfig:
    """Mimic Dinov2Backbone.config — only hidden_size is read downstream."""
    hidden_size: int = 768


class BioViLTVisionTower(nn.Module):
    """Drop-in replacement for MAIRA-2's Dinov2Backbone.

    Parameters
    ----------
    freeze_biovil : bool
        If True, BioViL-T encoder weights are frozen. The adapter Linear
        and the prefix token are always trainable.
    target_dim : int
        Output channel dim — must match MAIRA-2's projector input (768).
    """

    def __init__(self, freeze_biovil: bool = True, target_dim: int = 768,
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        try:
            from health_multimodal.image.utils import (
                get_image_inference, ImageModelType,
            )
        except ImportError as e:
            raise ImportError(
                "hi-ml-multimodal required for BioViL-T. "
                "pip install hi-ml-multimodal"
            ) from e

        engine = get_image_inference(ImageModelType.BIOVIL_T)
        self.biovil = engine.model.to(dtype)
        if freeze_biovil:
            for p in self.biovil.parameters():
                p.requires_grad_(False)

        # Adapter: project BioViL-T's 512-d patch tokens to MAIRA-2's 768-d.
        # Init with default Kaiming — magnitudes work out close to rad-DINO.
        self.adapter = nn.Linear(512, target_dim)
        # LayerNorm on the output to guarantee well-conditioned features
        # regardless of which image we encode. Without this, the occasional
        # BioViL-T feature with extreme values produced NaN logits in the LM
        # (asserts inside torch.multinomial). rad-DINO outputs are themselves
        # post-norm so this matches MAIRA-2's expected feature distribution.
        self.out_norm = nn.LayerNorm(target_dim)
        # Learned prefix token (replaces rad-DINO's CLS).
        self.prefix_token = nn.Parameter(torch.zeros(1, 1, target_dim))
        nn.init.normal_(self.prefix_token, std=0.02)

        # Buffers for de-/re-normalisation. Registered as buffers so they
        # move with .to(device).
        self.register_buffer(
            "_raddino_mean", torch.tensor(RADDINO_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_raddino_std", torch.tensor(RADDINO_STD).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )

        # Mimic Dinov2Backbone.config so downstream attribute access works.
        self.config = _VisionConfig(hidden_size=target_dim)

        # Cast adapter + prefix to requested dtype to match MAIRA-2 fp16 path.
        self.adapter = self.adapter.to(dtype)
        self.out_norm = self.out_norm.to(dtype)
        self.prefix_token.data = self.prefix_token.data.to(dtype)

        self._dtype = dtype

        logger.info(
            "BioViLTVisionTower ready (frozen biovil=%s, target_dim=%d, "
            "adapter+prefix trainable params=%d)",
            freeze_biovil, target_dim,
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def _renormalise(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Convert MAIRA-2-normalised pixels back to [0,1] then to ImageNet-norm,
        and resize from 518×518 to 448×448 (BioViL-T's training resolution)."""
        # Undo MAIRA-2 normalisation
        x = pixel_values.float()
        x = x * self._raddino_std.float() + self._raddino_mean.float()
        x = x.clamp(0, 1)
        # Resize to 448×448 if needed
        if x.shape[-1] != 448 or x.shape[-2] != 448:
            x = F.interpolate(x, size=(448, 448), mode="bilinear",
                              align_corners=False)
        # Re-normalise with ImageNet stats
        x = (x - self._imagenet_mean.float()) / self._imagenet_std.float()
        return x.to(self._dtype)

    def forward(self, pixel_values: torch.Tensor, **_ignored) -> _BackboneOutput:
        """pixel_values: (B, 3, 518, 518) — MAIRA-2's preprocessed tensor.

        Returns BackboneOutput with feature_maps=(features,) where
        features.shape == (B, 1370, 768).
        """
        x = self._renormalise(pixel_values)  # (B, 3, 448, 448)

        # BioViL-T expects a pair API but accepts previous_image=None.
        # Get just the per-image patch grid.
        encoder = self.biovil.encoder
        patch, _avg = encoder(
            current_image=x, previous_image=None,
            return_patch_embeddings=True,
        )
        # patch: (B, 512, 14, 14)

        # Bilinear-upsample to (B, 512, 37, 37)
        patch = F.interpolate(
            patch.float(), size=(TARGET_GRID, TARGET_GRID),
            mode="bilinear", align_corners=False,
        ).to(self._dtype)

        # Flatten to (B, 1369, 512)
        B, C, H, W = patch.shape
        tokens = patch.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Adapt to target_dim
        tokens = self.adapter(tokens)  # (B, 1369, 768)

        # Prepend the learned prefix token
        prefix = self.prefix_token.expand(B, -1, -1)  # (B, 1, 768)
        feat = torch.cat([prefix, tokens], dim=1)  # (B, 1370, 768)

        # LayerNorm to guarantee bounded features (and match the post-LN
        # statistics rad-DINO emits).
        feat = self.out_norm(feat)

        # Belt-and-braces: clamp any non-finite values that slipped through.
        # In practice this triggered when PEFT-wrapping interacted with our
        # custom forward; we drop NaN/Inf to zero so training can continue.
        if not torch.isfinite(feat).all():
            logger.warning("Non-finite values in BioViLTVisionTower output; zeroing them.")
            feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        # Hard clamp to fp16 safe range
        feat = feat.clamp(-32.0, 32.0)

        return _BackboneOutput(feature_maps=(feat,))
