"""Extract Google CXR Foundation Model features for Chest ImaGenome training pairs.

Google CXR is a TF SavedModel and cannot be fine-tuned end-to-end in PyTorch.
The equivalent of ImaGenome fine-tuning for Google CXR is:
  1. Extract frozen features for all ImaGenome train/val pairs (this script).
  2. Train a lightweight MLP probe on those features using train_google_cxr_imagenome.py.
  3. Evaluate on full MS-CXR-T test set (features extracted by extract_google_cxr_features.py).

Output layout:
  data/features/google_cxr_imagenome/
    {finding}_train.npy   — (N_train, 2*D) concatenated [prior, curr] features
    {finding}_val.npy     — (N_val, 2*D)
    {finding}_train_labels.npy
    {finding}_val_labels.npy

Usage:
    # In a TF-capable conda env:
    python scripts/extract_google_cxr_imagenome_features.py
    python scripts/extract_google_cxr_imagenome_features.py --finding edema
    python scripts/extract_google_cxr_imagenome_features.py --workers 4
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
    """Load the Google CXR elixr-c-v2-pooled SavedModel from HuggingFace."""
    try:
        import tensorflow as tf
        import tensorflow_text  # noqa: F401
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
    image_model_path = Path(model_dir) / "elixr-c-v2-pooled"
    logger.info("Loading SavedModel from %s", image_model_path)
    model = tf.saved_model.load(str(image_model_path))
    sigs = list(model.signatures.keys()) if hasattr(model, "signatures") else []
    logger.info("Loaded. Signatures: %s", sigs)
    infer_fn = model.signatures[sigs[0]] if sigs else model
    return infer_fn


def image_to_tf_example(path: Path) -> bytes:
    """Load a JPEG/PNG, resize to 1024×1024, serialise as tf.Example."""
    import tensorflow as tf

    img = Image.open(path).convert("L")
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
    """Run inference on one serialised tf.Example → (D,) float32 vector."""
    import tensorflow as tf

    tensor = tf.constant([serialised], dtype=tf.string)
    outputs = infer_fn(tensor)

    for key in ["feature_maps_0", "embedding", "embeddings", "output_0"]:
        if key in outputs:
            emb = outputs[key].numpy()
            break
    else:
        emb = list(outputs.values())[0].numpy()

    if emb.ndim == 4:
        return emb[0].mean(axis=(0, 1)).astype(np.float32)
    if emb.ndim == 3:
        return emb[0].mean(axis=0).astype(np.float32)
    return emb[0].astype(np.float32)


def extract_features_for_split(
    df, dicom_lookup: dict, infer_fn, split_name: str, finding: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract concatenated [prior, curr] features for all rows in df.

    Returns:
        pair_feats: (N, 2*D) float32
        labels:     (N,) int64  {0=improving, 1=stable, 2=worsening}
    """
    prior_feats, curr_feats = [], []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{finding}/{split_name}"):
        curr_path = dicom_lookup[row["curr_dicom_id"]]
        prior_path = dicom_lookup[row["prior_dicom_id"]]

        s_curr = image_to_tf_example(curr_path)
        s_prior = image_to_tf_example(prior_path)

        curr_feats.append(run_single(infer_fn, s_curr))
        prior_feats.append(run_single(infer_fn, s_prior))
        labels.append(int(row["label"]))

    pair_feats = np.concatenate(
        [np.stack(prior_feats), np.stack(curr_feats)], axis=1
    )  # (N, 2*D)
    return pair_feats, np.array(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs", default="data/imagenome_pairs/pairs_train.csv")
    parser.add_argument("--val_pairs", default="data/imagenome_pairs/pairs_val.csv")
    parser.add_argument("--dicom_to_path_csv", default="data/imagenome_pairs/dicom_to_path.csv")
    parser.add_argument("--imagenome_images", default="/data/imagenome_images")
    parser.add_argument("--out_dir", default="data/features/google_cxr_imagenome")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--finding", default=None, help="Extract only this finding (default: all)")
    args = parser.parse_args()

    import pandas as pd
    from data.dataset import FINDINGS
    from scripts.train_biovil_t_imagenome import build_dicom_lookup

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train_all = pd.read_csv(args.train_pairs)
    df_val_all = pd.read_csv(args.val_pairs)

    dicom_lookup = build_dicom_lookup(Path(args.imagenome_images), args.dicom_to_path_csv)
    logger.info("Image lookup: %d entries", len(dicom_lookup))

    infer_fn = load_tf_model(args.hf_cache_dir)

    findings = [args.finding] if args.finding else FINDINGS

    for finding in findings:
        train_feats_path = out_dir / f"{finding}_train.npy"
        train_labels_path = out_dir / f"{finding}_train_labels.npy"
        val_feats_path = out_dir / f"{finding}_val.npy"
        val_labels_path = out_dir / f"{finding}_val_labels.npy"

        if train_feats_path.exists() and val_feats_path.exists():
            logger.info("Cache exists, skipping: %s", finding)
            continue

        df_tr = df_train_all[df_train_all["finding"] == finding].copy()
        df_va = df_val_all[df_val_all["finding"] == finding].copy()

        # Filter to images we have
        df_tr = df_tr[df_tr["curr_dicom_id"].isin(dicom_lookup) & df_tr["prior_dicom_id"].isin(dicom_lookup)]
        df_va = df_va[df_va["curr_dicom_id"].isin(dicom_lookup) & df_va["prior_dicom_id"].isin(dicom_lookup)]

        if len(df_tr) == 0:
            logger.warning("No training pairs found for %s — skipping.", finding)
            continue

        logger.info("%s: %d train, %d val pairs", finding, len(df_tr), len(df_va))

        if not train_feats_path.exists():
            tr_feats, tr_labels = extract_features_for_split(df_tr, dicom_lookup, infer_fn, "train", finding)
            np.save(train_feats_path, tr_feats)
            np.save(train_labels_path, tr_labels)
            logger.info("Saved train: %s %s", train_feats_path, tr_feats.shape)

        if not val_feats_path.exists() and len(df_va) > 0:
            va_feats, va_labels = extract_features_for_split(df_va, dicom_lookup, infer_fn, "val", finding)
            np.save(val_feats_path, va_feats)
            np.save(val_labels_path, va_labels)
            logger.info("Saved val: %s %s", val_feats_path, va_feats.shape)

    logger.info("Done. Features saved to %s", out_dir)


if __name__ == "__main__":
    main()
