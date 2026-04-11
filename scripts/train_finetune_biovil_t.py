"""End-to-end fine-tuning of BioViL-T for temporal image classification.

Unlike the frozen-encoder approach in train.py, this script unfreezes the
BioViL-T temporal encoder and trains it jointly with a linear classification
head.  This replicates the likely evaluation protocol from Bannur et al. (CVPR
2023) Table 4, which appears to use fine-tuning rather than a frozen probe.

Protocol A setup (matching train.py):
  - Subject-level 70/10/20 split (loaded from data/splits/)
  - 3-class prediction: improving / stable / worsening
  - Metrics: macro-accuracy, macro-F1 per finding, averaged over 4 seeds
  - BioViL-T preprocessing: grayscale, resize=512, center_crop=448

Training details:
  - Differential LRs: backbone=1e-5, head=1e-3 (encoder frozen for warmup_epochs)
  - Class-weighted cross-entropy (Improving is underrepresented at ~18%)
  - Early stopping on val macro-accuracy (patience=15)
  - Checkpoints saved to checkpoints/biovil_t_finetune/

Usage:
    python scripts/train_finetune_biovil_t.py
    python scripts/train_finetune_biovil_t.py --seed 42 --finding edema
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
from torch.optim.lr_scheduler import CosineAnnealingLR
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

class BioViLTPairDataset(torch.utils.data.Dataset):
    """Loads (prior, current) image pairs with BioViL-T's own preprocessing."""

    def __init__(self, df, images_root: str, transform):
        self.df = df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from health_multimodal.image.data.io import load_image
        from data.dataset import dicom_id_to_filename

        row = self.df.iloc[idx]
        img1 = self.transform(load_image(self.images_root / dicom_id_to_filename(row["previous_dicom_id"])))
        img2 = self.transform(load_image(self.images_root / dicom_id_to_filename(row["dicom_id"])))
        return img1, img2, int(row["label"])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BioViLTClassifier(nn.Module):
    """BioViL-T temporal encoder + linear classification head."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        from health_multimodal.image.utils import get_image_inference, ImageModelType
        engine = get_image_inference(ImageModelType.BIOVIL_T)
        self.encoder_model = engine.model  # MultiImageModel (ResNet50 + ViT temporal pooler)
        self.head = nn.Linear(128, num_classes)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """img1=prior, img2=current.  Returns (B, num_classes) logits."""
        patch_fused, avg_pooled = self.encoder_model.encoder(
            current_image=img2,
            previous_image=img1,
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
# Training
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency class weights to handle Improving underrepresentation."""
    counts = np.bincount(y, minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img1, img2, labels in loader:
            logits = model(img1.to(device), img2.to(device))
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    macro_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return macro_acc, macro_f1


def train_one_finding(
    df_finding,
    split,
    finding,
    seed,
    images_root,
    device,
    epochs=50,
    warmup_epochs=5,
    batch_size=16,
    backbone_lr=1e-5,
    head_lr=1e-3,
    weight_decay=1e-4,
    patience=15,
    num_workers=4,
    ckpt_dir=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    subjects = df_finding["subject_id"].values

    def get_split(keys):
        mask = np.isin(subjects, keys)
        return df_finding[mask].reset_index(drop=True)

    df_train = get_split(split["train"])
    df_val = get_split(split["val"])
    df_test = get_split(split["test"])

    if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
        logger.warning("Empty split for %s seed=%d, skipping", finding, seed)
        return None

    from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
    transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=448)

    train_ds = BioViLTPairDataset(df_train, images_root, transform)
    val_ds = BioViLTPairDataset(df_val, images_root, transform)
    test_ds = BioViLTPairDataset(df_test, images_root, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Class weights from training labels
    class_weights = compute_class_weights(df_train["label"].values).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = BioViLTClassifier().to(device)

    # Phase 1: warm-up — freeze backbone, train only head
    for p in model.backbone_parameters():
        p.requires_grad_(False)

    opt_warmup = AdamW(model.head_parameters(), lr=head_lr, weight_decay=weight_decay)

    logger.info("[%s seed=%d] Warmup: %d epochs (head only)", finding, seed, warmup_epochs)
    for ep in range(warmup_epochs):
        model.train()
        for img1, img2, labels in train_loader:
            logits = model(img1.to(device), img2.to(device))
            loss = criterion(logits, labels.to(device))
            opt_warmup.zero_grad()
            loss.backward()
            opt_warmup.step()

    # Phase 2: full fine-tuning
    for p in model.backbone_parameters():
        p.requires_grad_(True)

    opt = AdamW([
        {"params": model.backbone_parameters(), "lr": backbone_lr},
        {"params": model.head_parameters(), "lr": head_lr},
    ], weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)

    best_val_acc = -1.0
    best_state = None
    no_improve = 0

    logger.info("[%s seed=%d] Fine-tuning: up to %d epochs", finding, seed, epochs)
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for img1, img2, labels in train_loader:
            logits = model(img1.to(device), img2.to(device))
            loss = criterion(logits, labels.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step()

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

        if (ep + 1) % 10 == 0:
            logger.info(
                "[%s seed=%d] ep=%d  loss=%.4f  val_macro_acc=%.3f  val_f1=%.3f  (best=%.3f)",
                finding, seed, ep + 1, total_loss / len(train_loader), val_acc, val_f1, best_val_acc,
            )

        if no_improve >= patience:
            logger.info("[%s seed=%d] Early stop at epoch %d", finding, seed, ep + 1)
            break

    # Evaluate best model on test set
    model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    logger.info(
        "[%s seed=%d] Test — macro_acc=%.3f  macro_f1=%.3f  (val_best=%.3f)",
        finding, seed, test_acc, test_f1, best_val_acc,
    )
    return {
        "finding": finding,
        "seed": seed,
        "macro_acc": test_acc,
        "macro_f1": test_f1,
        "best_val_macro_acc": best_val_acc,
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
    parser.add_argument("--labels", default="data/raw/ms_cxr_t_labels.csv")
    parser.add_argument("--images_root", default="data/raw/images")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789])
    parser.add_argument("--seed", type=int, default=None, help="Run a single seed")
    parser.add_argument("--finding", default=None, help="Run only this finding")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--ckpt_dir", default="checkpoints/biovil_t_finetune")
    args = parser.parse_args()

    from data.dataset import load_labels, FINDINGS

    seeds = [args.seed] if args.seed is not None else args.seeds
    findings = [args.finding] if args.finding else FINDINGS

    df = load_labels(args.labels, images_root=args.images_root)
    logger.info("Loaded %d rows across %d subjects", len(df), df["subject_id"].nunique())

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for finding in findings:
        df_f = df[df["finding"] == finding].copy()
        if len(df_f) == 0:
            continue

        for seed in seeds:
            split_path = Path("data/splits") / f"split_seed{seed}.json"
            if not split_path.exists():
                raise FileNotFoundError(f"Split not found: {split_path}. Run make_splits.py first.")
            with open(split_path) as f:
                split = json.load(f)

            result = train_one_finding(
                df_f, split, finding, seed,
                images_root=args.images_root,
                device=args.device,
                epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                batch_size=args.batch_size,
                backbone_lr=args.backbone_lr,
                head_lr=args.head_lr,
                patience=args.patience,
                ckpt_dir=args.ckpt_dir,
            )
            if result:
                all_results.append(result)

    if not all_results:
        logger.warning("No results produced.")
        return

    # Summarise
    import pandas as pd
    df_res = pd.DataFrame(all_results)
    logger.info("\n=== Summary ===")
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

    # Save
    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path("results") / f"biovil_t_finetune_protocolA_{timestamp}.json"
    payload = {
        "model": "biovil_t_finetune",
        "protocol": "A",
        "git_hash": git_hash(),
        "timestamp": timestamp,
        "seeds": seeds,
        "args": vars(args),
        "results": all_results,
        "summary": {
            finding: {
                "macro_acc": {"mean": float(g["macro_acc"].mean()), "std": float(g["macro_acc"].std())},
                "macro_f1": {"mean": float(g["macro_f1"].mean()), "std": float(g["macro_f1"].std())},
            }
            for finding, g in df_res.groupby("finding")
        },
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
