"""Extract SMB Vision v1 CXR frozen features for the full MS-CXR-T set.

Per-finding pair-concat features for the MLP probe to evaluate against, mirror
of extract_google_cxr_features.py but PyTorch-native using SMBVisionEncoderWrapper.

Output:
  data/features/smb_vision/{finding}_all.npy — (N_pairs, 2*2048) float32

Usage:
    python scripts/extract_smb_vision_features.py
    python scripts/extract_smb_vision_features.py --finding pleural_effusion
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
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
        img = load_image(self.paths[idx])
        return self.transform(img)


def build_val_transform(input_size: int = 448):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("L")),
        transforms.Resize(input_size + 64),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


@torch.no_grad()
def encode_paths(paths, model, device, batch_size, num_workers, input_size: int = 448):
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
            y = model(x)
        feats.append(y.float().cpu().numpy())
    return dict(zip(paths, np.concatenate(feats, axis=0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/raw/ms_cxr_t_labels.csv")
    parser.add_argument("--images_root", default="data/raw/images")
    parser.add_argument("--out_dir", default="data/features/smb_vision")
    parser.add_argument("--finding", default=None, help="Extract only this finding (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pooling_mode", choices=["merger", "deepstack_concat"], default="merger")
    parser.add_argument("--input_size", type=int, default=448)
    args = parser.parse_args()

    from data.dataset import load_labels, FINDINGS, dicom_id_to_filename
    from models.smb_vision import SMBVisionEncoderWrapper

    df = load_labels(args.labels, images_root=args.images_root)
    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    findings = [args.finding] if args.finding else FINDINGS
    pending = [f for f in findings if not (out_dir / f"{f}_all.npy").exists()]
    if not pending:
        logger.info("All caches present; nothing to do.")
        return

    # Gather every unique image path across pending findings (dedupe).
    unique_paths_set: set[Path] = set()
    for finding in pending:
        df_f = df[df["finding"] == finding]
        for col in ("previous_dicom_id", "dicom_id"):
            for did in df_f[col].tolist():
                unique_paths_set.add(images_root / dicom_id_to_filename(did))
    unique_paths = sorted(unique_paths_set, key=str)
    logger.info(
        "Pending findings: %s — %d unique images to encode",
        pending, len(unique_paths),
    )

    model = SMBVisionEncoderWrapper(
        gradient_checkpointing=False, pooling_mode=args.pooling_mode,
    ).to(args.device).eval()
    logger.info("SMB encoder loaded, pooling_mode=%s, embed_dim=%d", args.pooling_mode, model.embed_dim)

    embed_cache = encode_paths(
        unique_paths, model, args.device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        input_size=args.input_size,
    )
    logger.info("Encoded %d unique images.", len(embed_cache))

    del model
    torch.cuda.empty_cache()

    for finding in pending:
        df_f = df[df["finding"] == finding].reset_index(drop=True)
        prior = np.stack([
            embed_cache[images_root / dicom_id_to_filename(d)]
            for d in df_f["previous_dicom_id"]
        ], axis=0)
        curr = np.stack([
            embed_cache[images_root / dicom_id_to_filename(d)]
            for d in df_f["dicom_id"]
        ], axis=0)
        pair = np.concatenate([prior, curr], axis=1).astype(np.float32)
        np.save(out_dir / f"{finding}_all.npy", pair)
        logger.info("%s: saved %s", finding, pair.shape)

    logger.info("Done. Features in %s", out_dir)


if __name__ == "__main__":
    main()
