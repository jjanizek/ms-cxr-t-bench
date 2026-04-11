"""Stub for our foundation model.

This file will be populated when the model is ready for evaluation.
It must implement the BaseModel interface defined in models/base.py.
"""
import torch
from torch import Tensor

from models.base import BaseModel


class OurModel(BaseModel):
    """Placeholder for our in-house foundation model.

    TODO: Implement once model is available.
    """

    def __init__(self, checkpoint: str, device: str = "cpu"):
        raise NotImplementedError(
            "Our model is not yet implemented. "
            "Update this stub when the model is ready."
        )

    def encode_image(self, img: Tensor) -> Tensor:
        raise NotImplementedError

    def encode_image_pair(self, img1: Tensor, img2: Tensor) -> Tensor:
        raise NotImplementedError
