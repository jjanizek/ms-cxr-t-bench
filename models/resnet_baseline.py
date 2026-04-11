"""ImageNet-pretrained ResNet-50 frozen encoder baseline."""
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models

from models.base import BaseModel


class ResNetBaseline(BaseModel):
    """Frozen ResNet-50 (ImageNet weights) linear-probe baseline."""

    def __init__(self, freeze: bool = True):
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Drop the classification head; pool to (B, 2048)
        self._encoder = nn.Sequential(*list(backbone.children())[:-1])
        if freeze:
            for p in self._encoder.parameters():
                p.requires_grad_(False)
        self._encoder.eval()
        self._device = torch.device("cpu")

    def to(self, device):
        self._device = torch.device(device)
        self._encoder = self._encoder.to(self._device)
        return self

    def eval(self):
        self._encoder.eval()

    @torch.no_grad()
    def encode_image(self, img: Tensor) -> Tensor:
        """Returns (B, 2048) features."""
        img = img.to(self._device)
        feats = self._encoder(img)          # (B, 2048, 1, 1)
        return feats.flatten(1)

    def encode_image_pair(self, img1: Tensor, img2: Tensor) -> Tensor:
        """Concatenate per-image features → (B, 4096)."""
        f1 = self.encode_image(img1)
        f2 = self.encode_image(img2)
        return torch.cat([f1, f2], dim=1)
