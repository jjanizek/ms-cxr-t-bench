"""Google CXR Foundation Model wrapper.

The model (ELIXR) is TensorFlow/JAX only and is loaded from HuggingFace:
    google/cxr-foundation

Because it is TF-only, feature extraction must be done offline via:
    python scripts/extract_google_cxr_features.py

This wrapper supports two modes:
  1. Live inference (requires tensorflow + the HF model downloaded)
  2. Pre-extracted features loaded from data/features/google_cxr/ (recommended)

Embedding: mean-pooled over 32 patch tokens → 768-dim per image.
Pair representation: concatenation → 1536-dim.

References:
  - HuggingFace: https://huggingface.co/google/cxr-foundation
  - GitHub: https://github.com/Google-Health/cxr-foundation
"""
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from models.base import BaseModel

logger = logging.getLogger(__name__)

HF_MODEL_ID = "google/cxr-foundation"
EMBEDDING_DIM = 1376  # elixr-c-v2-pooled outputs (B, 8, 8, 1376); global avg pool → 1376-d


def preprocess_for_google_cxr(img_array: np.ndarray) -> bytes:
    """Convert a uint8/float numpy image (H, W) or (H, W, C) to
    a serialised tf.Example PNG suitable for the CXR Foundation model.

    The model expects:
      - Grayscale PNG serialised as bytes inside a tf.Example
      - Resolution 1024×1024 (we resize here)
      - No further normalisation — the model does z-score internally
    """
    import tensorflow as tf
    from PIL import Image

    # Convert to grayscale PIL
    if img_array.ndim == 3:
        pil_img = Image.fromarray(img_array).convert("L")
    else:
        pil_img = Image.fromarray(img_array)

    pil_img = pil_img.resize((1024, 1024), Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    example = tf.train.Example(features=tf.train.Features(feature={
        "image/encoded": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[png_bytes])
        )
    }))
    return example.SerializeToString()


class GoogleCXR(BaseModel):
    """Google CXR Foundation (ELIXR) image encoder.

    Args:
        hf_cache_dir: Directory where HuggingFace model is stored.
                      Pass None to use the default HF cache.
        device:       Ignored (model runs on TF; embeddings returned as
                      torch tensors on CPU, then moved by the training loop).
    """

    def __init__(
        self,
        hf_cache_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        self._device = torch.device(device)
        self._tf_model = None  # lazy load
        self._hf_cache_dir = hf_cache_dir
        self._load_model()

    def _load_model(self):
        try:
            import tensorflow as tf
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "tensorflow and huggingface_hub are required for GoogleCXR. "
                "Install with: pip install tensorflow huggingface_hub\n"
                "Alternatively, pre-extract features with:\n"
                "  python scripts/extract_google_cxr_features.py"
            ) from e

        logger.info("Downloading/loading CXR Foundation model from HuggingFace...")
        model_dir = snapshot_download(
            repo_id=HF_MODEL_ID,
            cache_dir=self._hf_cache_dir,
        )
        # The repo contains a TF SavedModel directory
        saved_model_path = Path(model_dir)
        self._tf_model = tf.saved_model.load(str(saved_model_path))

        # Discover the callable signature
        if hasattr(self._tf_model, "signatures"):
            sigs = list(self._tf_model.signatures.keys())
            logger.info("Available TF signatures: %s", sigs)
            if sigs:
                self._infer_fn = self._tf_model.signatures[sigs[0]]
            else:
                # Fall back to direct callable
                self._infer_fn = self._tf_model
        else:
            self._infer_fn = self._tf_model

        logger.info("GoogleCXR model loaded.")

    def to(self, device):
        # TF model doesn't use PyTorch device management
        self._device = torch.device(device)
        return self

    def eval(self):
        pass  # TF SavedModel is always in inference mode

    def _run_tf_inference(self, serialised_examples: list[bytes]) -> np.ndarray:
        """Run TF inference on a list of serialised tf.Example strings.

        Returns (N, 768) numpy array of mean-pooled patch embeddings.
        """
        import tensorflow as tf

        input_tensor = tf.constant(serialised_examples, dtype=tf.string)
        outputs = self._infer_fn(input_tensor)

        # Output key may vary by model version — try common names
        for key in ["embedding", "output_0", "embeddings"]:
            if key in outputs:
                emb = outputs[key].numpy()  # (N, 32, 768) or (N, 768)
                break
        else:
            # Take first output
            emb = list(outputs.values())[0].numpy()

        # Mean-pool over token dimension if needed
        if emb.ndim == 3:
            emb = emb.mean(axis=1)  # (N, 32, 768) → (N, 768)
        return emb.astype(np.float32)

    @torch.no_grad()
    def encode_image(self, img: Tensor) -> Tensor:
        """Encode a batch of images.

        Args:
            img: (B, C, H, W) float tensor in [0,1] (ImageNet-normalised or raw).
                 The model does its own normalisation internally; we just need
                 valid pixel values.

        Returns:
            (B, 768) feature tensor.
        """
        # Denormalise from ImageNet stats back to [0, 255] uint8 for PNG encoding
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        img_raw = ((img * std + mean) * 255).clamp(0, 255).byte().cpu().numpy()

        serialised = []
        for i in range(img_raw.shape[0]):
            arr = img_raw[i].transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
            serialised.append(preprocess_for_google_cxr(arr))

        emb = self._run_tf_inference(serialised)  # (B, 768)
        return torch.from_numpy(emb).to(self._device)

    def encode_image_pair(self, img1: Tensor, img2: Tensor) -> Tensor:
        """Concatenate embeddings → (B, 1536)."""
        f1 = self.encode_image(img1)
        f2 = self.encode_image(img2)
        return torch.cat([f1, f2], dim=1)
