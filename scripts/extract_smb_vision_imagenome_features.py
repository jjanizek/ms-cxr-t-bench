"""Extract SMB Vision v1 CXR frozen features for Chest ImaGenome pairs.

The end-to-end fine-tune on this 600M model collapsed for 2/5 findings (edema,
pneumothorax) and underperformed on the others — likely a combination of head
saddle-point issues and over-parameterised backbone updates wrecking the
pretrained features. The frozen-encoder + MLP probe protocol (same as we use
for Google CXR, which can't be fine-tuned end-to-end) preserves the pretrained
representations and lets the MLP do the small amount of supervised work.

Output layout (matches Google CXR):
  data/features/smb_vision_imagenome/
    {finding}_train.npy        — (N_train, 2*2048) concat [prior, curr]
    {finding}_train_labels.npy — (N_train,) int64
    {finding}_val.npy          — (N_val, 2*2048)
    {finding}_val_labels.npy   — (N_val,) int64

Image-level features are deduped (each unique dicom_id encoded once) before
pair concatenation — a single dicom may appear in many pairs.

Usage:
    python scripts/extract_smb_vision_imagenome_features.py
    python scripts/extract_smb_vision_imagenome_features.py --finding edema
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from health_multimodal.image.data.io import load_image
        img = load_image(self.paths[idx])  # PIL 'L'
        return self.transform(img)


def build_val_transform(input_size: int = 448):
    """Same val transform as the end-to-end fine-tune (grayscale, 0.5/0.5).

    Resize to input_size + 64, center-crop to input_size — matches the
    canonical 512→448 ratio when input_size==448.
    """
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("L")),
        transforms.Resize(input_size + 64),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


@torch.no_grad()
def encode_paths(paths: list[str], model, device: str, batch_size: int, num_workers: int, input_size: int = 448) -> dict[str, np.ndarray]:
    """Encode each unique path once → dict{path -> (D,) float32}."""
    transform = build_val_transform(input_size=input_size)
    ds = ImagePathDataset(paths, transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    feats = []
    for x in tqdm(loader, desc="encode"):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y = model(x)  # (B, D)
        feats.append(y.float().cpu().numpy())
    arr = np.concatenate(feats, axis=0)  # (N_unique, D)
    return dict(zip(paths, arr))


def assemble_pair_features(df: pd.DataFrame, dicom_lookup: dict, embed_cache: dict) -> tuple[np.ndarray, np.ndarray]:
    prior = np.stack([embed_cache[dicom_lookup[d]] for d in df["prior_dicom_id"]], axis=0)
    curr = np.stack([embed_cache[dicom_lookup[d]] for d in df["curr_dicom_id"]], axis=0)
    pair = np.concatenate([prior, curr], axis=1).astype(np.float32)
    labels = df["label"].values.astype(np.int64)
    return pair, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs", default="data/imagenome_pairs/pairs_train.csv")
    parser.add_argument("--val_pairs", default="data/imagenome_pairs/pairs_val.csv")
    parser.add_argument("--dicom_to_path_csv", default="data/imagenome_pairs/dicom_to_path.csv")
    parser.add_argument("--imagenome_images", default="/data/mimic-cxr-jpg")
    parser.add_argument("--out_dir", default="data/features/smb_vision_imagenome")
    parser.add_argument("--finding", default=None, help="Extract only this finding (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pooling_mode", choices=["merger", "deepstack_concat"], default="merger",
                        help="Encoder pooling. 'merger' = final merger output (2048-d). "
                             "'deepstack_concat' = final + 3 deepstack levels (8192-d).")
    parser.add_argument("--input_size", type=int, default=448,
                        help="Image input crop size (must be multiple of 32 = patch*spatial_merge).")
    args = parser.parse_args()

    from data.dataset import FINDINGS
    from scripts.train_biovil_t_imagenome import build_dicom_lookup
    from models.smb_vision import SMBVisionEncoderWrapper

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train_all = pd.read_csv(args.train_pairs)
    df_val_all = pd.read_csv(args.val_pairs)
    dicom_lookup = build_dicom_lookup(Path(args.imagenome_images), args.dicom_to_path_csv)
    logger.info("Image lookup: %d entries", len(dicom_lookup))

    findings = [args.finding] if args.finding else FINDINGS

    # Filter pairs to those whose images exist
    def filter_pairs(df):
        return df[df["curr_dicom_id"].isin(dicom_lookup) & df["prior_dicom_id"].isin(dicom_lookup)].copy()

    df_train_all = filter_pairs(df_train_all)
    df_val_all = filter_pairs(df_val_all)

    # If a finding's caches already exist, skip it for path-collection too.
    pending_findings = []
    for finding in findings:
        tr_p = out_dir / f"{finding}_train.npy"
        va_p = out_dir / f"{finding}_val.npy"
        if tr_p.exists() and va_p.exists():
            logger.info("Cache exists, skipping: %s", finding)
            continue
        pending_findings.append(finding)
    if not pending_findings:
        logger.info("Nothing to do.")
        return

    # Collect unique image paths across all pending findings.
    needed_dicom_ids: set[str] = set()
    for finding in pending_findings:
        for df in (df_train_all, df_val_all):
            sub = df[df["finding"] == finding]
            needed_dicom_ids.update(sub["prior_dicom_id"].tolist())
            needed_dicom_ids.update(sub["curr_dicom_id"].tolist())
    unique_paths = sorted({dicom_lookup[d] for d in needed_dicom_ids})
    logger.info(
        "Pending findings: %s — %d unique images to encode",
        pending_findings, len(unique_paths),
    )

    # Load model once
    model = SMBVisionEncoderWrapper(
        gradient_checkpointing=False, pooling_mode=args.pooling_mode,
    ).to(args.device).eval()
    logger.info(
        "SMB encoder loaded on %s, pooling_mode=%s, embed_dim=%d",
        args.device, args.pooling_mode, model.embed_dim,
    )

    embed_cache = encode_paths(
        unique_paths, model, args.device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        input_size=args.input_size,
    )
    logger.info("Encoded %d unique images.", len(embed_cache))

    # Free GPU memory before disk writes
    del model
    torch.cuda.empty_cache()

    for finding in pending_findings:
        df_tr = df_train_all[df_train_all["finding"] == finding]
        df_va = df_val_all[df_val_all["finding"] == finding]

        if len(df_tr) == 0:
            logger.warning("No train pairs for %s — skipping.", finding)
            continue

        tr_feats, tr_labels = assemble_pair_features(df_tr, dicom_lookup, embed_cache)
        np.save(out_dir / f"{finding}_train.npy", tr_feats)
        np.save(out_dir / f"{finding}_train_labels.npy", tr_labels)
        logger.info("%s train: %s saved", finding, tr_feats.shape)

        if len(df_va) > 0:
            va_feats, va_labels = assemble_pair_features(df_va, dicom_lookup, embed_cache)
            np.save(out_dir / f"{finding}_val.npy", va_feats)
            np.save(out_dir / f"{finding}_val_labels.npy", va_labels)
            logger.info("%s val: %s saved", finding, va_feats.shape)

    logger.info("Done. Features in %s", out_dir)


if __name__ == "__main__":
    main()
