"""Wrapper for Standard Model Biomedicine's SMB Vision v1 CXR encoder.

HuggingFace model: standardmodelbio/smb-vision-v1-cxr

Architecture notes:
- Qwen2.5-VL-style ragged-token vision transformer (27 layers, hidden 1152,
  600M params), grayscale (in_channels=1), patch_size=16, temporal_patch_size=16.
- Input is (N_total_patches, raw_dim) pixels + grid_thw of shape (B, 3) with
  rows [T_token=1, H/16, W/16]; there is no conventional (B, C, H, W) batched
  forward.
- Encoder returns (N_total_patches, 1152). The `merger` submodule (trained)
  does a spatial-merge-size=2 shuffle + MLP → (N/4, 2048). We apply the merger
  then mean-pool per image to a single 2048-dim embedding.
- The model file imports `transformers.modeling_layers.GradientCheckpointingLayer`
  which first appears in transformers 4.53. We shim it for the 4.49 env used
  by this repo (upgrading risks breaking hi-ml-multimodal / BioViL-T).
"""

from __future__ import annotations

import sys
import types
from typing import Optional

import torch
import torch.nn as nn


def _shim_transformers_for_smb() -> None:
    """Inject symbols the SMB modeling file expects (present in transformers
    >=4.53/4.57 but not in the 4.49 env used by this repo).

    - transformers.modeling_layers.GradientCheckpointingLayer
    - transformers.utils.TransformersKwargs  (typing-only alias → TypedDict)
    """
    import transformers

    try:
        from transformers.modeling_layers import GradientCheckpointingLayer  # noqa: F401
    except ImportError:
        class GradientCheckpointingLayer(nn.Module):
            gradient_checkpointing = False

            def __call__(self, *args, **kwargs):
                if self.gradient_checkpointing and self.training:
                    return torch.utils.checkpoint.checkpoint(
                        super().__call__, *args, use_reentrant=False, **kwargs
                    )
                return super().__call__(*args, **kwargs)

        mod = types.ModuleType("transformers.modeling_layers")
        mod.GradientCheckpointingLayer = GradientCheckpointingLayer
        sys.modules["transformers.modeling_layers"] = mod
        transformers.modeling_layers = mod

    try:
        from transformers.utils import TransformersKwargs  # noqa: F401
    except ImportError:
        from typing import TypedDict

        class TransformersKwargs(TypedDict, total=False):
            pass

        import transformers.utils as _utils
        _utils.TransformersKwargs = TransformersKwargs


_shim_transformers_for_smb()


class SMBVisionEncoderWrapper(nn.Module):
    """Wraps the SMB vision encoder to expose a standard (B,1,H,W) -> (B,D) API.

    Pooling modes:
      - "merger" (default): run the trained spatial-merge=2 merger on the final
        layer's tokens, mean-pool → (B, 2048).
      - "deepstack_concat": mean-pool the 3 trained deepstack mergers' outputs
        (taken from layers in vision_config.deepstack_visual_indexes — typically
        [8, 16, 24]) plus the final merger output, then concatenate → (B, 4*2048).
        Captures multi-resolution features.

    H and W must be multiples of (patch_size * spatial_merge_size) = 32 so the
    merger's reshape is well-defined.
    """

    def __init__(
        self,
        hf_id: str = "standardmodelbio/smb-vision-v1-cxr",
        attn_implementation: str = "sdpa",
        gradient_checkpointing: bool = False,
        torch_dtype: torch.dtype = torch.float32,
        pooling_mode: str = "merger",
    ):
        super().__init__()
        # AutoModel's loader trips over `config_class` being None on the
        # dynamically-resolved class in transformers 4.49. Resolve the classes
        # directly and pass an explicit config to bypass that path.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        ConfigCls = get_class_from_dynamic_module(
            "configuration_smb_vision.SMBVisionModelConfig", hf_id
        )
        ModelCls = get_class_from_dynamic_module(
            "modeling_smb_vision.SMBVisionModel", hf_id
        )
        cfg = ConfigCls.from_pretrained(hf_id)
        full = ModelCls.from_pretrained(
            hf_id,
            config=cfg,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        # Keep only the vision encoder (drop the MIM/JEPA predictor + auxiliary
        # heads that were only used during self-supervised pretraining).
        self.encoder = full.encoder
        del full

        self.patch_size = self.encoder.config.patch_size
        self.temporal_patch_size = self.encoder.config.temporal_patch_size
        self.spatial_merge_size = self.encoder.config.spatial_merge_size
        self.hidden_size = self.encoder.config.hidden_size
        self.out_hidden_size = self.encoder.config.out_hidden_size

        if gradient_checkpointing:
            for blk in self.encoder.blocks:
                blk.gradient_checkpointing = True

        if pooling_mode not in ("merger", "deepstack_concat"):
            raise ValueError(f"Unknown pooling_mode: {pooling_mode}")
        self.pooling_mode = pooling_mode
        self.num_deepstack_levels = len(self.encoder.config.deepstack_visual_indexes)

    @property
    def embed_dim(self) -> int:
        if self.pooling_mode == "deepstack_concat":
            # final merger + N deepstack mergers, all out_hidden_size each
            return self.out_hidden_size * (1 + self.num_deepstack_levels)
        return self.out_hidden_size  # 2048

    def _patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 1, H, W) grayscale float tensor → (pixels_flat, grid_thw).

        pixels_flat: (B * h * w, 1 * tps * P * P)
        grid_thw:    (B, 3) with each row [1, h, w].
        """
        B, C, H, W = x.shape
        assert C == 1, f"Expected 1-channel grayscale input, got {C} channels"
        P = self.patch_size
        TPS = self.temporal_patch_size
        assert H % (P * self.spatial_merge_size) == 0 and W % (P * self.spatial_merge_size) == 0, (
            f"H={H}, W={W} must be multiples of patch_size*spatial_merge_size="
            f"{P*self.spatial_merge_size}"
        )
        h, w = H // P, W // P

        # (B, 1, h, w, P, P)
        patches = x.unfold(2, P, P).unfold(3, P, P)
        # (B, h*w, 1, P, P)
        patches = patches.contiguous().view(B, 1, h * w, P, P).permute(0, 2, 1, 3, 4)
        # tile temporal: (B, h*w, 1, TPS, P, P) — all frames identical for a still image
        patches = patches.unsqueeze(3).expand(B, h * w, 1, TPS, P, P).contiguous()
        # flatten to (B*h*w, 1*TPS*P*P)
        pixel_values = patches.view(B * h * w, -1)

        grid_thw = torch.tensor(
            [[1, h, w]] * B, dtype=torch.long, device=x.device
        )
        return pixel_values, grid_thw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) → (B, out_hidden_size=2048).

        Accepts an RGB (B, 3, H, W) input too — it is averaged to grayscale,
        which costs nothing when the three channels are a replicated grayscale
        image.
        """
        if x.dim() == 4 and x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)

        # The encoder's pos-embed interpolation calls .tolist() on CPU, so build
        # grid_thw on CPU then move to the encoder's device when used inside.
        pixel_values, grid_thw = self._patchify(x)
        # Match encoder's parameter dtype (patch_embed conv weights).
        pixel_values = pixel_values.to(self.encoder.patch_embed.proj_c1.weight.dtype)
        hidden_states, deepstack_features = self.encoder(pixel_values, grid_thw)
        # hidden_states: (B*h*w, 1152). Apply trained merger → (B*h*w/4, 2048).
        merged = self.encoder.merger(hidden_states)
        B = x.shape[0]
        D = self.out_hidden_size
        merged = merged.view(B, -1, D)             # (B, h*w/4, 2048)
        pooled_final = merged.mean(dim=1)          # (B, 2048)

        if self.pooling_mode == "merger":
            return pooled_final

        # deepstack_concat: pool each level then concat.
        # deepstack_features is a list of (B*h*w/4, 2048) tensors (already merged).
        levels = [pooled_final]
        for ds in deepstack_features:
            ds_pooled = ds.view(B, -1, D).mean(dim=1)  # (B, 2048)
            levels.append(ds_pooled)
        return torch.cat(levels, dim=1)            # (B, (1+L)*2048)
