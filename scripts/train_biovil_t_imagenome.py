"""Train BioViL-T on Chest ImaGenome, evaluate on MS-CXR-T.

Replicates the BioViL-T paper (Bannur et al., CVPR 2023) Table 2 evaluation:
  - Training data: Chest ImaGenome silver temporal pairs (pairs_train.csv)
  - Validation:    ImaGenome val split (pairs_val.csv)
  - Test:          All 1,326 MS-CXR-T pairs (the fixed held-out test set)
  - Model:         Full end-to-end fine-tuning of BioViL-T + MLP head
  - Protocol:      NOT subject-level split of MS-CXR-T — uses entire MS-CXR-T as test

This is distinct from Protocol A (train/val/test split of MS-CXR-T itself).

Paper hyperparameters (Bannur et al., Appendix):
  - 30 epochs, LR 1e-5, AdamW, batch 128
  - No class weighting mentioned; we add optional class weighting (--class_weight)
  - Augmentation: random flips, crops, affine, color jitter, Gaussian noise

Usage:
    python scripts/train_biovil_t_imagenome.py
    python scripts/train_biovil_t_imagenome.py --finding edema --seed 42
"""
import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dicom_lookup(mimic_root: Path, dicom_to_path_csv: str = None) -> dict:
    """Build dicom_id → absolute path mapping.

    Supports three layouts:
      1. Flat:     mimic_root/{dicom_id}.jpg
      2. gsutil:   mimic_root/files/p*/p*/s*/{dicom_id}.jpg
                   (gsutil rsync gs://physionet-open/mimic-cxr-jpg/2.1.0 {mimic_root})
      3. wget -r:  mimic_root/physionet.org/files/mimic-cxr-jpg/2.1.0/files/p*/...

    If dicom_to_path_csv is provided, uses the pre-computed relative path mapping
    from extract_imagenome_pairs.py to avoid scanning the full tree.
    """
    import pandas as pd

    lookup = {}

    if dicom_to_path_csv and Path(dicom_to_path_csv).exists():
        df = pd.read_csv(dicom_to_path_csv)
        wget_prefix = mimic_root / "physionet.org" / "files" / "mimic-cxr-jpg" / "2.1.0"
        for _, row in df.iterrows():
            dicom_id = row["dicom_id"]
            rel_path = row["rel_path"]  # e.g. files/p10/p10000032/s50414267/02aa804e-...jpg
            if not rel_path:
                continue
            # Try layouts in order: gsutil → wget-r → flat
            for candidate in (
                mimic_root / rel_path,          # gsutil rsync layout
                wget_prefix / rel_path,         # wget -r layout
                mimic_root / f"{dicom_id}.jpg", # flat layout
            ):
                if candidate.exists():
                    lookup[dicom_id] = candidate
                    break
        return lookup

    # No CSV — fall back to flat layout
    for jpg in mimic_root.glob("*.jpg"):
        lookup[jpg.stem] = jpg
    return lookup


class ImaGenomePairDataset(torch.utils.data.Dataset):
    """(prior, current) pairs from Chest ImaGenome with optional augmentation."""

    def __init__(self, df, dicom_lookup: dict, transform):
        self.df = df.reset_index(drop=True)
        self.dicom_lookup = dicom_lookup
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from health_multimodal.image.data.io import load_image
        row = self.df.iloc[idx]
        curr_path = self.dicom_lookup[row["curr_dicom_id"]]
        prior_path = self.dicom_lookup[row["prior_dicom_id"]]
        img_curr_pil = load_image(curr_path)
        img_prior_pil = load_image(prior_path)
        # Paper: "synchronise image data augmentations to apply identical transforms
        # to the current and prior images" — fix RNG state so both get same spatial aug.
        seed = torch.randint(0, 2**31, (1,)).item()
        torch.manual_seed(seed)
        img_curr = self.transform(img_curr_pil)
        torch.manual_seed(seed)
        img_prior = self.transform(img_prior_pil)
        return img_prior, img_curr, int(row["label"])


class MSCXRTPairDataset(torch.utils.data.Dataset):
    """MS-CXR-T pairs for test evaluation."""

    def __init__(self, df, images_root: str, transform):
        from data.dataset import dicom_id_to_filename
        self.df = df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.transform = transform
        self.dicom_id_to_filename = dicom_id_to_filename

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from health_multimodal.image.data.io import load_image
        row = self.df.iloc[idx]
        curr_path = self.images_root / self.dicom_id_to_filename(row["dicom_id"])
        prior_path = self.images_root / self.dicom_id_to_filename(row["previous_dicom_id"])
        img_curr = self.transform(load_image(curr_path))
        img_prior = self.transform(load_image(prior_path))
        return img_prior, img_curr, int(row["label"])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BioViLTClassifier(nn.Module):
    """BioViL-T temporal encoder + MLP classification head.

    Head: 128 → 64 → 3, matching the paper's MLP description.
    """

    def __init__(self, num_classes: int = 3, hidden: int = 64):
        super().__init__()
        from health_multimodal.image.utils import get_image_inference, ImageModelType
        engine = get_image_inference(ImageModelType.BIOVIL_T)
        self.encoder_model = engine.model
        self.head = nn.Sequential(
            nn.Linear(128, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, img_prior: torch.Tensor, img_curr: torch.Tensor) -> torch.Tensor:
        patch_fused, avg_pooled = self.encoder_model.encoder(
            current_image=img_curr,
            previous_image=img_prior,
            return_patch_embeddings=True,
        )
        out = self.encoder_model.forward_post_encoder(patch_fused, avg_pooled)
        emb = F.normalize(out.projected_global_embedding, dim=-1)  # (B, 128)
        return self.head(emb)

    def backbone_parameters(self):
        return self.encoder_model.parameters()

    def head_parameters(self):
        return self.head.parameters()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, return_logits=False):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for img_prior, img_curr, labels in loader:
            logits = model(img_prior.to(device), img_curr.to(device))
            all_logits.append(logits.float().cpu().numpy())
            all_labels.extend(labels.numpy())
    logits_arr = np.concatenate(all_logits, axis=0)
    y_pred = logits_arr.argmax(axis=1)
    y_true = np.array(all_labels)
    macro_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if return_logits:
        return macro_acc, macro_f1, logits_arr, y_true
    return macro_acc, macro_f1


def train_one_finding(
    df_train, df_val, df_test,
    finding, seed, imagenome_lookup, mscxrt_images_root,
    device, epochs=30, warmup_epochs=3, batch_size=32,
    grad_accum_steps=4, use_amp=True,
    backbone_lr=1e-5, head_lr=1e-3, weight_decay=1e-4,
    patience=10, num_workers=8, class_weight=True, ckpt_dir=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    from torchvision import transforms
    from health_multimodal.image.data.transforms import (
        ExpandChannels,
        create_chest_xray_transform_for_inference,
    )

    # Training augmentation matching paper (Bannur et al., Appendix F).
    # Match inference preprocessing (no ImageNet normalize; ExpandChannels 1→3).
    train_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=30, shear=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        ExpandChannels(),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])
    val_transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=448)

    train_ds = ImaGenomePairDataset(df_train, imagenome_lookup, train_transform)
    val_ds = ImaGenomePairDataset(df_val, imagenome_lookup, val_transform)
    test_ds = MSCXRTPairDataset(df_test, mscxrt_images_root, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    criterion_weight = None
    if class_weight:
        criterion_weight = compute_class_weights(df_train["label"].values).to(device)
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)

    model = BioViLTClassifier().to(device)
    amp_dtype = torch.float16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Phase 1: warmup — head only
    for p in model.backbone_parameters():
        p.requires_grad_(False)
    opt_warmup = AdamW(model.head_parameters(), lr=head_lr, weight_decay=weight_decay)
    logger.info("[%s seed=%d] Warmup: %d epochs", finding, seed, warmup_epochs)
    for _ in range(warmup_epochs):
        model.train()
        for img_prior, img_curr, labels in train_loader:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(img_prior.to(device), img_curr.to(device))
                loss = criterion(logits, labels.to(device))
            opt_warmup.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt_warmup)
            scaler.update()

    # Phase 2: full fine-tuning
    for p in model.backbone_parameters():
        p.requires_grad_(True)

    # Exempt positional encodings and missing-image embeddings from weight decay,
    # matching paper ("as in [73]").
    no_decay_names = {"position", "pos_embed", "missing_image"}
    backbone_decay, backbone_no_decay = [], []
    for name, param in model.encoder_model.named_parameters():
        if any(nd in name for nd in no_decay_names):
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)

    opt = AdamW([
        {"params": backbone_decay,        "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": backbone_no_decay,     "lr": backbone_lr, "weight_decay": 0.0},
        {"params": model.head_parameters(), "lr": head_lr,  "weight_decay": weight_decay},
    ])

    # Linear LR schedule: linear warmup for warmup_proportion of total steps,
    # then linear decay to 0. Paper: warmup_proportion=0.03, 30 epochs.
    steps_per_epoch = max(1, len(train_loader) // grad_accum_steps)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_epochs / epochs)  # warmup_epochs≈0.03*30=~1

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(opt, lr_lambda)

    best_val_acc = -1.0
    best_state = None
    no_improve = 0

    logger.info("[%s seed=%d] Fine-tuning: %d epochs", finding, seed, epochs)
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        opt.zero_grad()
        for step, (img_prior, img_curr, labels) in enumerate(
            tqdm(train_loader, desc=f"ep{ep+1}", leave=False)
        ):
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(img_prior.to(device), img_curr.to(device))
                loss = criterion(logits, labels.to(device)) / grad_accum_steps
            scaler.scale(loss).backward()
            total_loss += loss.item() * grad_accum_steps
            if (step + 1) % grad_accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                opt.zero_grad()

        val_acc, val_f1 = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if ckpt_dir:
                ckpt_path = Path(ckpt_dir) / f"{finding}_seed{seed}_best.pt"
                torch.save(best_state, ckpt_path)
        else:
            no_improve += 1

        logger.info(
            "[%s seed=%d] ep=%d/%d  loss=%.4f  val_acc=%.3f  val_f1=%.3f  (best=%.3f)",
            finding, seed, ep + 1, epochs,
            total_loss / len(train_loader), val_acc, val_f1, best_val_acc,
        )

        if no_improve >= patience:
            logger.info("[%s seed=%d] Early stop at epoch %d", finding, seed, ep + 1)
            break

    # Test on MS-CXR-T (save per-sample logits + predictions to avoid ever
    # needing to re-eval from checkpoints).
    model.load_state_dict(best_state)
    test_acc, test_f1, test_logits, test_y_true = evaluate(
        model, test_loader, device, return_logits=True,
    )
    logger.info(
        "[%s seed=%d] MS-CXR-T test — macro_acc=%.3f  macro_f1=%.3f  (best_val=%.3f)",
        finding, seed, test_acc, test_f1, best_val_acc,
    )

    preds_dir = Path("results/predictions/biovil_t_imagenome")
    preds_dir.mkdir(parents=True, exist_ok=True)
    label_names = ["improving", "stable", "worsening"]
    preds_df = df_test.reset_index(drop=True).copy()
    preds_df = preds_df.assign(
        gt_label=test_y_true,
        ground_truth=[label_names[int(y)] for y in test_y_true],
        pred_label=test_logits.argmax(axis=1),
        predicted=[label_names[int(p)] for p in test_logits.argmax(axis=1)],
        logit_improving=test_logits[:, 0],
        logit_stable=test_logits[:, 1],
        logit_worsening=test_logits[:, 2],
    )
    preds_path = preds_dir / f"{finding}_seed{seed}_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    logger.info("  saved per-sample predictions → %s", preds_path)

    return {
        "finding": finding,
        "seed": seed,
        "macro_acc": test_acc,
        "macro_f1": test_f1,
        "best_val_macro_acc": best_val_acc,
        "predictions_csv": str(preds_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs", default="data/imagenome_pairs/pairs_train.csv")
    parser.add_argument("--val_pairs", default="data/imagenome_pairs/pairs_val.csv")
    parser.add_argument("--dicom_to_path_csv", default="data/imagenome_pairs/dicom_to_path.csv",
                        help="CSV mapping dicom_id→rel_path (from extract_imagenome_pairs.py)")
    parser.add_argument("--mscxrt_labels", default="data/raw/ms_cxr_t_labels.csv")
    # Accept either flat (/data/imagenome_images) or wget-r hierarchy (/data/mimic-cxr-jpg)
    parser.add_argument("--imagenome_images", default="/data/mimic-cxr-jpg",
                        help="Root of MIMIC-CXR-JPG images (flat or wget -r hierarchy)")
    parser.add_argument("--mscxrt_images", default="data/raw/images")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--finding", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-step batch size. Effective batch = batch_size * grad_accum_steps.")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Gradient accumulation (default 4 × batch 32 = paper effective 128).")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed-precision training.")
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no_class_weight", action="store_true")
    parser.add_argument("--ckpt_dir", default="checkpoints/biovil_t_imagenome")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    import pandas as pd
    from data.dataset import load_labels, FINDINGS

    seeds = [args.seed] if args.seed is not None else args.seeds
    findings = [args.finding] if args.finding else FINDINGS

    # Load training / val data from ImaGenome
    df_train_all = pd.read_csv(args.train_pairs)
    df_val_all = pd.read_csv(args.val_pairs)
    logger.info(
        "ImaGenome train: %d pairs, val: %d pairs", len(df_train_all), len(df_val_all)
    )

    # Build dicom_id → path lookup (handles both flat and wget-r hierarchy)
    logger.info("Building image lookup from %s ...", args.imagenome_images)
    imagenome_lookup = build_dicom_lookup(
        Path(args.imagenome_images), args.dicom_to_path_csv
    )
    logger.info("Image lookup: %d entries", len(imagenome_lookup))

    # Load MS-CXR-T (full dataset as fixed test set)
    df_test_all = load_labels(args.mscxrt_labels, images_root=args.mscxrt_images)
    logger.info("MS-CXR-T test: %d pairs", len(df_test_all))

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for finding in findings:
        df_tr = df_train_all[df_train_all["finding"] == finding].copy()
        df_va = df_val_all[df_val_all["finding"] == finding].copy()
        df_te = df_test_all[df_test_all["finding"] == finding].copy()

        # Filter to pairs where both images are available
        df_tr = df_tr[
            df_tr["curr_dicom_id"].isin(imagenome_lookup) &
            df_tr["prior_dicom_id"].isin(imagenome_lookup)
        ].copy()
        df_va = df_va[
            df_va["curr_dicom_id"].isin(imagenome_lookup) &
            df_va["prior_dicom_id"].isin(imagenome_lookup)
        ].copy()

        if len(df_tr) == 0:
            logger.warning("No training pairs with downloaded images for %s, skipping", finding)
            continue
        if len(df_te) == 0:
            logger.warning("No MS-CXR-T test pairs for %s, skipping", finding)
            continue

        logger.info(
            "Finding: %s — train=%d, val=%d, test(MS-CXR-T)=%d",
            finding, len(df_tr), len(df_va), len(df_te),
        )

        for seed in seeds:
            result = train_one_finding(
                df_tr, df_va, df_te,
                finding=finding, seed=seed,
                imagenome_lookup=imagenome_lookup,
                mscxrt_images_root=args.mscxrt_images,
                device=args.device,
                epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                use_amp=not args.no_amp,
                backbone_lr=args.backbone_lr,
                head_lr=args.head_lr,
                patience=args.patience,
                class_weight=not args.no_class_weight,
                ckpt_dir=str(ckpt_dir),
                num_workers=args.num_workers,
            )
            if result:
                all_results.append(result)

    if not all_results:
        logger.warning("No results produced.")
        return

    import pandas as pd
    df_res = pd.DataFrame(all_results)
    logger.info("\n=== Summary (MS-CXR-T test set) ===")
    for finding in findings:
        sub = df_res[df_res["finding"] == finding]
        if len(sub) == 0:
            continue
        logger.info(
            "  %-20s  macro_acc=%.3f±%.3f  macro_f1=%.3f±%.3f",
            finding,
            sub["macro_acc"].mean(), sub["macro_acc"].std(),
            sub["macro_f1"].mean(), sub["macro_f1"].std(),
        )
    per_seed = df_res.groupby("seed")["macro_acc"].mean()
    logger.info(
        "  %-20s  macro_acc=%.3f±%.3f",
        "AVERAGE", per_seed.mean(), per_seed.std(),
    )

    # Save results
    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path("results") / f"biovil_t_imagenome_{timestamp}.json"
    payload = {
        "model": "biovil_t_imagenome",
        "protocol": "paper_replication",
        "description": "Trained on ImaGenome silver, tested on full MS-CXR-T (paper Table 2 replication)",
        "git_hash": git_hash(),
        "timestamp": timestamp,
        "seeds": seeds,
        "args": vars(args),
        "results": all_results,
        "summary": {
            finding: {
                "macro_acc": {"mean": float(g["macro_acc"].mean()), "std": float(g["macro_acc"].std())},
                "macro_f1":  {"mean": float(g["macro_f1"].mean()),  "std": float(g["macro_f1"].std())},
            }
            for finding, g in df_res.groupby("finding")
        },
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
