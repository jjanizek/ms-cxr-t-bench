"""Offline feature extraction for BioViL-T.

Uses the hi-ml-multimodal ImageInferenceEngine to load images from disk with
BioViL-T's own preprocessing (resize=512, center_crop=448, grayscale), then
calls the MultiImageEncoder temporal forward to produce a single 128-dim
l2-normalised embedding per (prior, current) image pair.

Features are saved to data/features/biovil_t/ as (N, 128) .npy files,
one per finding.

Usage:
    python scripts/extract_biovil_t_features.py
    python scripts/extract_biovil_t_features.py --device cuda
    python scripts/extract_biovil_t_features.py --finding edema
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_engine(device: str):
    from health_multimodal.image.utils import get_image_inference, ImageModelType
    logger.info("Loading BioViL-T model...")
    engine = get_image_inference(ImageModelType.BIOVIL_T)
    engine.model = engine.model.to(device).eval()
    return engine


@torch.no_grad()
def encode_pair(engine, path1: Path, path2: Path, device: str) -> np.ndarray:
    """Encode a (prior, current) pair → (128,) l2-normalised numpy array."""
    model = engine.model
    transform = engine.transform

    img1, _ = engine.load_and_transform_input_image(path1, transform)  # (1, 1, H, W)
    img2, _ = engine.load_and_transform_input_image(path2, transform)  # (1, 1, H, W)
    img1 = img1.to(device)
    img2 = img2.to(device)

    # Temporal forward: prior=img1, current=img2
    patch_fused, avg_pooled = model.encoder(
        current_image=img2,
        previous_image=img1,
        return_patch_embeddings=True,
    )
    out = model.forward_post_encoder(patch_fused, avg_pooled)
    emb = F.normalize(out.projected_global_embedding, dim=-1)  # (1, 128)
    return emb[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/raw/ms_cxr_t_labels.csv")
    parser.add_argument("--images_root", default="data/raw/images")
    parser.add_argument("--out_dir", default="data/features/biovil_t")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--finding", default=None, help="Extract only this finding")
    args = parser.parse_args()

    from data.dataset import load_labels, FINDINGS, dicom_id_to_filename

    df = load_labels(args.labels, images_root=args.images_root)
    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s", args.device)
    engine = load_engine(args.device)

    findings = [args.finding] if args.finding else FINDINGS

    for finding in findings:
        cache_path = out_dir / f"{finding}_all.npy"
        if cache_path.exists():
            logger.info("Cache exists, skipping: %s", cache_path)
            continue

        df_f = df[df["finding"] == finding].reset_index(drop=True)
        n = len(df_f)
        logger.info("Extracting %s (%d pairs)...", finding, n)

        results = []
        for _, row in tqdm(df_f.iterrows(), total=n, desc=finding):
            path_prior = images_root / dicom_id_to_filename(row["previous_dicom_id"])
            path_curr = images_root / dicom_id_to_filename(row["dicom_id"])
            results.append(encode_pair(engine, path_prior, path_curr, args.device))

        feats = np.stack(results, axis=0)  # (N, 128)
        np.save(cache_path, feats)
        logger.info("Saved %s: shape %s", cache_path, feats.shape)

    logger.info("Done.")


if __name__ == "__main__":
    main()
