"""Offline feature extraction for the Google CXR Foundation Model.

Runs TensorFlow inference on all image pairs in the MS-CXR-T dataset and
saves per-finding .npy feature caches to data/features/google_cxr/.

This script is meant to be run once in an environment that has TensorFlow
installed.  The main train.py pipeline then loads these cached features
directly without needing TF.

Usage:
    # In a TF-capable conda env (e.g. with tensorflow installed):
    python scripts/extract_google_cxr_features.py
    python scripts/extract_google_cxr_features.py --hf_cache_dir /data/hf_cache
    python scripts/extract_google_cxr_features.py --batch_size 16 --finding pleural_effusion
"""
import argparse
import io
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_tf_model(hf_cache_dir: str | None):
    try:
        import tensorflow as tf
        import tensorflow_text  # noqa: F401 — registers SentencepieceOp and other custom ops
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "tensorflow, tensorflow-text, and huggingface_hub are required.\n"
            "Install: pip install tensorflow tensorflow-text huggingface_hub"
        )
        sys.exit(1)

    logger.info("Loading google/cxr-foundation from HuggingFace...")
    model_dir = snapshot_download(
        repo_id="google/cxr-foundation",
        cache_dir=hf_cache_dir,
    )
    # The HF repo has two subdirs: elixr-c-v2-pooled (image) and pax-elixr-b-text (text).
    # We want the image encoder.
    image_model_path = Path(model_dir) / "elixr-c-v2-pooled"
    logger.info("Loading SavedModel from %s", image_model_path)
    model = tf.saved_model.load(str(image_model_path))
    sigs = list(model.signatures.keys()) if hasattr(model, "signatures") else []
    logger.info("Loaded. Signatures: %s", sigs)
    infer_fn = model.signatures[sigs[0]] if sigs else model
    return infer_fn


def image_to_tf_example(path: Path) -> bytes:
    """Load a JPEG/PNG from disk, resize to 1024×1024, serialise as tf.Example."""
    import tensorflow as tf

    img = Image.open(path).convert("L")        # grayscale
    img = img.resize((1024, 1024), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    example = tf.train.Example(features=tf.train.Features(feature={
        "image/encoded": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[png_bytes])
        )
    }))
    return example.SerializeToString()


def run_single(infer_fn, serialised: bytes) -> np.ndarray:
    """Run inference on a single serialised tf.Example.

    The elixr-c-v2-pooled model (TF1 SavedModel) processes one image at a time.
    Output key 'feature_maps_0' has shape (1, 8, 8, 1376).
    We global-average-pool over the 8×8 spatial grid → (1376,).
    """
    import tensorflow as tf

    tensor = tf.constant([serialised], dtype=tf.string)
    outputs = infer_fn(tensor)

    for key in ["feature_maps_0", "embedding", "embeddings", "output_0"]:
        if key in outputs:
            emb = outputs[key].numpy()   # (1, 8, 8, 1376) or similar
            break
    else:
        emb = list(outputs.values())[0].numpy()

    # (1, H, W, C) → (C,)
    if emb.ndim == 4:
        return emb[0].mean(axis=(0, 1)).astype(np.float32)
    # (1, T, C) → (C,)
    if emb.ndim == 3:
        return emb[0].mean(axis=0).astype(np.float32)
    # (1, C) → (C,)
    return emb[0].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/raw/ms_cxr_t_labels.csv")
    parser.add_argument("--images_root", default="data/raw/images")
    parser.add_argument("--out_dir", default="data/features/google_cxr")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument(
        "--finding", default=None,
        help="Extract only this finding (default: all)"
    )
    args = parser.parse_args()

    from data.dataset import load_labels, FINDINGS, dicom_id_to_filename

    df = load_labels(args.labels, images_root=args.images_root)
    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    infer_fn = load_tf_model(args.hf_cache_dir)

    findings = [args.finding] if args.finding else FINDINGS

    for finding in findings:
        cache_path = out_dir / f"{finding}_all.npy"
        if cache_path.exists():
            logger.info("Cache exists, skipping: %s", cache_path)
            continue

        df_f = df[df["finding"] == finding].reset_index(drop=True)
        n = len(df_f)
        logger.info("Extracting %s (%d pairs)...", finding, n)

        # Extract features for prior and current images separately, then concatenate.
        # Embedding dim is discovered from the first batch (don't assume 768).
        def extract_all(dicom_col: str) -> np.ndarray:
            results = []
            for _, row in tqdm(df_f.iterrows(), total=n, desc=f"{finding}/{dicom_col}"):
                s = image_to_tf_example(images_root / dicom_id_to_filename(row[dicom_col]))
                results.append(run_single(infer_fn, s))
            return np.stack(results, axis=0)  # (N, 1376)

        prior_feats = extract_all("previous_dicom_id")
        curr_feats = extract_all("dicom_id")
        logger.info("Embedding dim: %d (per image)", prior_feats.shape[1])

        # Concatenate prior + current → (N, 2*D)
        pair_feats = np.concatenate([prior_feats, curr_feats], axis=1)
        np.save(cache_path, pair_feats)
        logger.info("Saved %s: shape %s", cache_path, pair_feats.shape)

    logger.info("Done.")


if __name__ == "__main__":
    main()
