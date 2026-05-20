"""Vision tower that swaps MAIRA-2's rad-DINO for an ENSEMBLE of fine-tuned
BioViL-T encoders (one per finding), trained on Chest ImaGenome and known to
hit 0.612 macro_acc on MS-CXR-T end-to-end.

The hypothesis: features from BioViL-T that has been *fine-tuned* for chest-
X-ray temporal classification are intrinsically much more informative than
features from generic pretrained BioViL-T (or frozen rad-DINO). Concatenating
across the 5 finding-specific encoders gives 2560-d patch features per token
that span all 5 MS-CXR-T findings.

Encoders are frozen. The trainable parts are:
  - adapter Linear(2560 → 768)
  - learned prefix token (768-d)
  - output LayerNorm

This module behaves like a Dinov2Backbone replacement for MAIRA-2's
multi_modal_projector: returns BackboneOutput(feature_maps=(x,)) where
x.shape == (B, 1370, 768). See biovil_t_vision_tower.py for the simpler
single-model variant.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# rad-DINO normalisation used by MAIRA-2's processor (so we can undo it).
RADDINO_MEAN = (0.5307, 0.5307, 0.5307)
RADDINO_STD = (0.2583, 0.2583, 0.2583)

# BioViL-T expects ImageNet stats at 448×448.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TARGET_GRID = 37
TARGET_TOKENS = TARGET_GRID * TARGET_GRID + 1  # 1370


@dataclass
class _BackboneOutput:
    feature_maps: tuple


@dataclass
class _VisionConfig:
    hidden_size: int = 768


class BioViLTEnsembleVisionTower(nn.Module):
    """Ensemble of fine-tuned BioViL-T encoders → adapter → MAIRA-2-compatible features.

    Parameters
    ----------
    finding_ckpts : dict[str, str]
        Map finding name (e.g. 'consolidation') to checkpoint path.
        All 5 checkpoints share the BioViL-T architecture; weights differ.
    target_dim : int
        Output channel dim for MAIRA-2's projector input (768).
    freeze_encoders : bool
        If True (default), all encoder params are frozen. The adapter +
        prefix + out_norm are always trainable.
    """

    def __init__(
        self,
        finding_ckpts: dict,
        target_dim: int = 768,
        freeze_encoders: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        try:
            from health_multimodal.image.utils import (
                get_image_inference, ImageModelType,
            )
        except ImportError as e:
            raise ImportError(
                "hi-ml-multimodal required. pip install hi-ml-multimodal"
            ) from e

        self.findings = list(finding_ckpts.keys())
        self.encoders = nn.ModuleList()
        for finding in self.findings:
            engine = get_image_inference(ImageModelType.BIOVIL_T)
            enc = engine.model  # full BioViL-T (encoder + projector + head missing)
            # Load fine-tuned weights, stripping 'encoder_model.' prefix.
            ckpt = torch.load(finding_ckpts[finding], map_location="cpu")
            new_sd = {}
            for k, v in ckpt.items():
                if k.startswith("encoder_model."):
                    new_sd[k.replace("encoder_model.", "")] = v
            missing, unexpected = enc.load_state_dict(new_sd, strict=False)
            if unexpected:
                logger.warning("%s: unexpected keys %d (head left untouched)",
                               finding, len(unexpected))
            if missing:
                logger.warning("%s: missing keys %d", finding, len(missing))
            enc = enc.to(dtype)
            if freeze_encoders:
                for p in enc.parameters():
                    p.requires_grad_(False)
            self.encoders.append(enc)
            logger.info("Loaded fine-tuned BioViL-T for %s from %s",
                        finding, finding_ckpts[finding])

        n_findings = len(self.findings)
        in_dim = 512 * n_findings  # 2560 for 5 findings

        # Adapter: concat-of-5 (512 each) → 768
        self.adapter = nn.Linear(in_dim, target_dim)
        # Output LayerNorm — same rationale as the single-model wrapper.
        self.out_norm = nn.LayerNorm(target_dim)
        # Learned prefix token
        self.prefix_token = nn.Parameter(torch.zeros(1, 1, target_dim))
        nn.init.normal_(self.prefix_token, std=0.02)

        # Buffers for de-/re-normalisation
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

        self.config = _VisionConfig(hidden_size=target_dim)

        self.adapter = self.adapter.to(dtype)
        self.out_norm = self.out_norm.to(dtype)
        self.prefix_token.data = self.prefix_token.data.to(dtype)
        self._dtype = dtype

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "BioViLTEnsembleVisionTower ready (%d encoders, in_dim=%d, "
            "target_dim=%d, trainable=%d)",
            n_findings, in_dim, target_dim, trainable,
        )

    def _renormalise(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values.float()
        x = x * self._raddino_std.float() + self._raddino_mean.float()
        x = x.clamp(0, 1)
        if x.shape[-1] != 448 or x.shape[-2] != 448:
            x = F.interpolate(x, size=(448, 448), mode="bilinear",
                              align_corners=False)
        x = (x - self._imagenet_mean.float()) / self._imagenet_std.float()
        return x.to(self._dtype)

    def _encode_paired(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch with each finding model in paired mode if possible.

        x: (B, 3, 448, 448) where we expect pairs stacked along batch dim,
        i.e. images come in (a0, b0, a1, b1, ...) order with each (a_i, b_i)
        a (current, prior) pair from one MAIRA-2 prompt. We use each model
        with the *partner* image as the previous-image context so both
        positions get temporally-aware features (order-agnostic — same
        compute either way).

        If B is odd or 1, fall back to single-image mode (placeholder prev).
        Returns: per-model patch features (B, 512, 14, 14) — concat happens
        outside.
        """
        B = x.shape[0]
        per_model_features = []
        for enc_model in self.encoders:
            if B >= 2 and B % 2 == 0:
                # Reshape into pairs of 2
                paired = x.view(B // 2, 2, *x.shape[1:])  # (N, 2, 3, H, W)
                # Encode each image with the other as the prior. Two calls,
                # roles flipped each time.
                tokens_a, _ = enc_model.encoder(
                    current_image=paired[:, 0],
                    previous_image=paired[:, 1],
                    return_patch_embeddings=True,
                )
                tokens_b, _ = enc_model.encoder(
                    current_image=paired[:, 1],
                    previous_image=paired[:, 0],
                    return_patch_embeddings=True,
                )
                # Re-interleave back to (B, 512, 14, 14) matching input order
                stacked = torch.stack([tokens_a, tokens_b], dim=1)
                patch = stacked.view(B, *tokens_a.shape[1:])
            else:
                # Odd or single → single-image mode with placeholder previous
                patch, _ = enc_model.encoder(
                    current_image=x, previous_image=None,
                    return_patch_embeddings=True,
                )
            per_model_features.append(patch)
        return per_model_features

    def forward(self, pixel_values: torch.Tensor, **_ignored) -> _BackboneOutput:
        x = self._renormalise(pixel_values)  # (B, 3, 448, 448)

        patches = self._encode_paired(x)
        # Concat along channel dim → (B, 2560, 14, 14)
        feat = torch.cat(patches, dim=1)

        # Bilinear upsample to (B, 2560, 37, 37), then flatten spatially
        feat = F.interpolate(
            feat.float(), size=(TARGET_GRID, TARGET_GRID),
            mode="bilinear", align_corners=False,
        ).to(self._dtype)
        B, C, H, W = feat.shape
        tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 1369, 2560)

        # Adapter
        tokens = self.adapter(tokens)  # (B, 1369, 768)

        # Prepend prefix
        prefix = self.prefix_token.expand(B, -1, -1)
        out = torch.cat([prefix, tokens], dim=1)  # (B, 1370, 768)

        # LayerNorm + finite-safety clamp (lessons from v7 NaN debugging)
        out = self.out_norm(out)
        if not torch.isfinite(out).all():
            logger.warning("Non-finite values in ensemble output; zeroing them.")
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = out.clamp(-32.0, 32.0)

        return _BackboneOutput(feature_maps=(out,))
