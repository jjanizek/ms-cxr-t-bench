"""Re-evaluate saved checkpoints on MS-CXR-T, dumping per-sample predictions.

Writes a CSV with columns:
    dicom_id, previous_dicom_id, finding, ground_truth, gt_label,
    predicted, pred_label, logit_improving, logit_stable, logit_worsening

These CSVs are the format expected by scripts/compute_contradiction_rate.py.

Supported models (seed 42 only — that's what we have checkpoints for):
    biovil_t_imagenome          checkpoints/biovil_t_imagenome/{finding}_seed42_best.pt
    google_cxr_imagenome        checkpoints/google_cxr_imagenome/google_cxr_{finding}_seed42_best.pt
    imagenet_densenet121_attn   checkpoints/imagenome_generic/imagenet_densenet121_{finding}_seed42_best.pt
                                (the DenseNet concat checkpoints were overwritten by the attention run)

Usage:
    python scripts/reeval_predictions.py --model biovil_t_imagenome
    python scripts/reeval_predictions.py --model google_cxr_imagenome
    python scripts/reeval_predictions.py --model imagenet_densenet121_attn
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LABEL_NAMES = ["improving", "stable", "worsening"]


# ---------------------------------------------------------------------------
# Common: MS-CXR-T test data
# ---------------------------------------------------------------------------

def load_mscxrt(labels_csv: str, images_root: str):
    from data.dataset import load_labels, FINDINGS
    df = load_labels(labels_csv, images_root=images_root)
    logger.info("MS-CXR-T: %d (pair, finding) rows", len(df))
    return df, FINDINGS


# ---------------------------------------------------------------------------
# BioViL-T
# ---------------------------------------------------------------------------

def eval_biovil_t(args, df, findings):
    from scripts.train_biovil_t_imagenome import BioViLTClassifier, MSCXRTPairDataset
    from health_multimodal.image.data.transforms import (
        create_chest_xray_transform_for_inference,
    )

    val_transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=448)
    device = args.device
    ckpt_dir = Path(args.ckpt_dir or "checkpoints/biovil_t_imagenome")

    rows = []
    for finding in findings:
        df_f = df[df["finding"] == finding].reset_index(drop=True)
        if len(df_f) == 0:
            continue
        ds = MSCXRTPairDataset(df_f, args.mscxrt_images, val_transform)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True)

        ckpt_path = ckpt_dir / f"{finding}_seed{args.seed}_best.pt"
        logger.info("[%s] loading %s", finding, ckpt_path)
        model = BioViLTClassifier().to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        all_logits = []
        with torch.inference_mode():
            for img_prior, img_curr, labels in tqdm(loader, desc=finding, leave=False):
                logits = model(img_prior.to(device), img_curr.to(device))
                all_logits.append(logits.float().cpu().numpy())
        logits_arr = np.concatenate(all_logits, axis=0)
        rows.extend(_build_rows(df_f, logits_arr, finding))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Google CXR (frozen features → MLP probe)
# ---------------------------------------------------------------------------

def eval_google_cxr(args, df, findings):
    from scripts.train_google_cxr_imagenome import MLPProbe

    device = args.device
    ckpt_dir = Path(args.ckpt_dir or "checkpoints/google_cxr_imagenome")
    mscxrt_features_dir = Path(args.mscxrt_features_dir
                               or "data/features/google_cxr")

    rows = []
    for finding in findings:
        df_f = df[df["finding"] == finding].reset_index(drop=True)
        if len(df_f) == 0:
            continue

        feats_path = mscxrt_features_dir / f"{finding}_all.npy"
        if not feats_path.exists():
            logger.warning("missing features: %s — skipping %s", feats_path, finding)
            continue
        feats = np.load(feats_path)
        if len(df_f) != len(feats):
            raise ValueError(
                f"{finding}: df has {len(df_f)} rows but features has {len(feats)}"
            )

        ckpt_path = ckpt_dir / f"google_cxr_{finding}_seed{args.seed}_best.pt"
        logger.info("[%s] loading %s", finding, ckpt_path)
        model = MLPProbe(in_dim=feats.shape[1]).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        with torch.inference_mode():
            X = torch.from_numpy(feats).float().to(device)
            logits_arr = model(X).float().cpu().numpy()
        rows.extend(_build_rows(df_f, logits_arr, finding))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DenseNet-121 with cross-attention head
# ---------------------------------------------------------------------------

def eval_densenet_attn(args, df, findings):
    from scripts.train_finetune_imagenome_generic import (
        load_encoder, CrossAttnPairClassifier, MSCXRTPairDataset,
    )
    from torchvision import transforms

    device = args.device
    ckpt_dir = Path(args.ckpt_dir or "checkpoints/imagenome_generic")

    to_rgb = transforms.Lambda(lambda img: img.convert("RGB"))
    val_transform = transforms.Compose([
        to_rgb,
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    rows = []
    for finding in findings:
        df_f = df[df["finding"] == finding].reset_index(drop=True)
        if len(df_f) == 0:
            continue
        ds = MSCXRTPairDataset(df_f, args.mscxrt_images, val_transform)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True)

        ckpt_path = ckpt_dir / f"imagenet_densenet121_{finding}_seed{args.seed}_best.pt"
        logger.info("[%s] loading %s", finding, ckpt_path)
        encoder, embed_dim = load_encoder("imagenet_densenet121", device)
        model = CrossAttnPairClassifier(encoder, embed_dim).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        all_logits = []
        with torch.inference_mode():
            for img_prior, img_curr, labels in tqdm(loader, desc=finding, leave=False):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device != "cpu")):
                    logits = model(img_prior.to(device), img_curr.to(device))
                all_logits.append(logits.float().cpu().numpy())
        logits_arr = np.concatenate(all_logits, axis=0)
        rows.extend(_build_rows(df_f, logits_arr, finding))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_rows(df_f: pd.DataFrame, logits_arr: np.ndarray, finding: str):
    pred_labels = logits_arr.argmax(axis=1)
    rows = []
    for i, row in df_f.iterrows():
        rows.append({
            "dicom_id": row["dicom_id"],
            "previous_dicom_id": row["previous_dicom_id"],
            "finding": finding,
            "ground_truth": row["progression"],
            "gt_label": int(row["label"]),
            "predicted": LABEL_NAMES[int(pred_labels[i])],
            "pred_label": int(pred_labels[i]),
            "logit_improving": float(logits_arr[i, 0]),
            "logit_stable": float(logits_arr[i, 1]),
            "logit_worsening": float(logits_arr[i, 2]),
        })
    return rows


def print_summary(df: pd.DataFrame, findings):
    accs = []
    for finding in findings:
        sub = df[df["finding"] == finding]
        if len(sub) == 0:
            continue
        acc = balanced_accuracy_score(sub["gt_label"], sub["pred_label"])
        f1 = f1_score(sub["gt_label"], sub["pred_label"], average="macro", zero_division=0)
        accs.append(acc)
        logger.info("  %-20s  n=%-4d  macro_acc=%.3f  macro_f1=%.3f", finding, len(sub), acc, f1)
    logger.info("  %-20s           macro_acc=%.3f (average of findings)", "AVERAGE",
                float(np.mean(accs)) if accs else float("nan"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EVAL_FNS = {
    "biovil_t_imagenome": eval_biovil_t,
    "google_cxr_imagenome": eval_google_cxr,
    "imagenet_densenet121_attn": eval_densenet_attn,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(EVAL_FNS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mscxrt_labels", default="data/raw/ms_cxr_t_labels.csv")
    parser.add_argument("--mscxrt_images", default="data/raw/images")
    parser.add_argument("--mscxrt_features_dir", default="data/features/google_cxr")
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="results/reeval")
    args = parser.parse_args()

    df, findings = load_mscxrt(args.mscxrt_labels, args.mscxrt_images)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_seed{args.seed}_predictions.csv"

    preds_df = EVAL_FNS[args.model](args, df, findings)
    preds_df.to_csv(out_path, index=False)
    logger.info("Saved %d predictions to %s", len(preds_df), out_path)

    logger.info("=== Summary ===")
    print_summary(preds_df, findings)


if __name__ == "__main__":
    main()
