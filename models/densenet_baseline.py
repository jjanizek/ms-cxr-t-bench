"""ImageNet-pretrained DenseNet-121 frozen encoder baseline.

DenseNet-121 is the standard architecture in CXR literature (CheXNet, etc.).
Frozen encoder + linear probe via Protocol A.
Output: (B, 1024) per image, (B, 2048) concatenated pair.
"""
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models

from models.base import BaseModel


class DenseNetBaseline(BaseModel):
    """Frozen DenseNet-121 (ImageNet weights) linear-probe baseline."""

    def __init__(self, freeze: bool = True):
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        # DenseNet: features → relu → adaptive_avg_pool → flatten → classifier
        # We keep everything up to the pool and discard the classifier.
        self._features = backbone.features  # nn.Sequential of DenseBlocks
        self._pool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze:
            for p in self._features.parameters():
                p.requires_grad_(False)
        self._features.eval()

        self._device = torch.device("cpu")

    def to(self, device):
        self._device = torch.device(device)
        self._features = self._features.to(self._device)
        self._pool = self._pool.to(self._device)
        return self

    def eval(self):
        self._features.eval()

    @torch.no_grad()
    def encode_image(self, img: Tensor) -> Tensor:
        """Returns (B, 1024) features."""
        img = img.to(self._device)
        feat_maps = self._features(img)           # (B, 1024, H, W)
        feat_maps = torch.relu(feat_maps)         # DenseNet uses relu after features
        pooled = self._pool(feat_maps)            # (B, 1024, 1, 1)
        return pooled.flatten(1)                  # (B, 1024)

    def encode_image_pair(self, img1: Tensor, img2: Tensor) -> Tensor:
        """Concatenate per-image features → (B, 2048)."""
        f1 = self.encode_image(img1)
        f2 = self.encode_image(img2)
        return torch.cat([f1, f2], dim=1)
