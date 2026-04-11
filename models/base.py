"""Base model interface for MS-CXR-T benchmarking."""
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class BaseModel(ABC):
    """Abstract base class all benchmark models must implement.

    The minimal surface is encode_image / encode_image_pair.
    encode_text is optional — only vision-language models need it.
    """

    @abstractmethod
    def encode_image(self, img: Tensor) -> Tensor:
        """Extract features from a single image.

        Args:
            img: (B, C, H, W) pre-processed image tensor.

        Returns:
            (B, D) feature tensor.
        """

    @abstractmethod
    def encode_image_pair(self, img1: Tensor, img2: Tensor) -> Tensor:
        """Encode a temporal image pair into a single feature vector.

        Default implementation: concatenate encode_image outputs.
        Temporal models (e.g. BioViL-T) may override with a joint encoder.

        Args:
            img1: (B, C, H, W) earlier study image.
            img2: (B, C, H, W) later study image.

        Returns:
            (B, 2*D) or (B, D) feature tensor (depending on model).
        """

    def encode_text(self, text: list[str]) -> Optional[Tensor]:
        """Encode text strings. Optional — return None if not supported.

        Args:
            text: List of B strings.

        Returns:
            (B, D) feature tensor, or None.
        """
        return None

    def eval(self):
        """Put model in eval mode (no-op for non-nn.Module wrappers)."""

    def to(self, device):
        """Move model to device (no-op for non-nn.Module wrappers)."""
        return self
