"""End-to-end fine-tuning on Chest ImaGenome for any PyTorch encoder model.

Intended for our own foundation model (and any other PyTorch-based encoder).
For BioViL-T use train_biovil_t_imagenome.py (joint temporal encoder, different arch).
For Google CXR use extract_google_cxr_imagenome_features.py + train.py (TF model,
  cannot be fine-tuned end-to-end — frozen probe on ImaGenome features only).

Architecture for static encoders:
  encode_image(prior)  → f_prior  (D-dim)
  encode_image(current) → f_curr  (D-dim)
  concat([f_prior, f_curr]) → 2D-dim → MLP → 3-class logits

Training matches BioViL-T paper protocol where possible:
  - 30 epochs, linear warmup 0.03, base LR 1e-5, AdamW, batch 128
  - Weighted cross entropy
  - Synchronized augmentation across prior/current images
  - Augmentation: resize 512, random crop 448, flip, affine, color jitter, noise

Usage:
    python scripts/train_finetune_imagenome_generic.py --model ours
    python scripts/train_finetune_imagenome_generic.py --model ours --finding edema --seed 42
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
# Dataset (identical to train_biovil_t_imagenome — synchronized augmentation)
# ---------------------------------------------------------------------------

class ImaGenomePairDataset(torch.utils.data.Dataset):
    def __init__(self, df, dicom_lookup: dict, transform):
        self.df = df.reset_index(drop=True)
        self.dicom_lookup = dicom_lookup
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from health_multimodal.image.data.io import load_image
        row = self.df.iloc[idx]
        curr_pil = load_image(self.dicom_lookup[row["curr_dicom_id"]])
        prior_pil = load_image(self.dicom_lookup[row["prior_dicom_id"]])
        seed = torch.randint(0, 2**31, (1,)).item()
        torch.manual_seed(seed)
        img_curr = self.transform(curr_pil)
        torch.manual_seed(seed)
        img_prior = self.transform(prior_pil)
        return img_prior, img_curr, int(row["label"])


class MSCXRTPairDataset(torch.utils.data.Dataset):
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
        curr_pil = load_image(self.images_root / self.dicom_id_to_filename(row["dicom_id"]))
        prior_pil = load_image(self.images_root / self.dicom_id_to_filename(row["previous_dicom_id"]))
        seed = torch.randint(0, 2**31, (1,)).item()
        torch.manual_seed(seed)
        img_curr = self.transform(curr_pil)
        torch.manual_seed(seed)
        img_prior = self.transform(prior_pil)
        return img_prior, img_curr, int(row["label"])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConcatPairClassifier(nn.Module):
    """Static encoder (any PyTorch model) + concat pair embedding + MLP head.

    encode_image(prior) and encode_image(curr) are called separately,
    their embeddings concatenated, then passed through a 2-layer MLP.
    """

    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int = 3, hidden: int = 256):
        super().__init__()
        self.encoder = encoder  # must implement forward(img) → (B, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, img_prior: torch.Tensor, img_curr: torch.Tensor) -> torch.Tensor:
        f_prior = self.encoder(img_prior)   # (B, D)
        f_curr = self.encoder(img_curr)     # (B, D)
        pair = torch.cat([f_prior, f_curr], dim=1)  # (B, 2D)
        return self.head(pair)

    def backbone_parameters(self):
        return self.encoder.parameters()

    def head_parameters(self):
        return self.head.parameters()


def load_encoder(model_name: str, device: str):
    """Load a PyTorch encoder by name. Returns (encoder_module, embed_dim).

    Add new models here as they become available.
    """
    if model_name == "ours":
        # TODO: replace with actual model import once available
        raise NotImplementedError(
            "Our foundation model is not yet integrated. "
            "Add it to models/ours.py and update this function."
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"For BioViL-T use train_biovil_t_imagenome.py. "
            f"For Google CXR use extract_google_cxr_imagenome_features.py + train.py."
        )


# ---------------------------------------------------------------------------
# Training helpers (identical to train_biovil_t_imagenome)
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img_prior, img_curr, labels in loader:
            logits = model(img_prior.to(device), img_curr.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    y_pred, y_true = np.array(all_preds), np.array(all_labels)
    return float(balanced_accuracy_score(y_true, y_pred)), float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def train_one_finding(
    df_train, df_val, df_test,
    finding, seed, model_name, imagenome_lookup, mscxrt_images_root,
    device, epochs=30, warmup_epochs=1, batch_size=128,
    backbone_lr=1e-5, head_lr=1e-3, weight_decay=1e-4,
    patience=10, num_workers=8, class_weight=True, ckpt_dir=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    from torchvision import transforms
    from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference

    train_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=30, shear=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])
    val_transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=448)

    train_ds = ImaGenomePairDataset(df_train, imagenome_lookup, train_transform)
    val_ds = ImaGenomePairDataset(df_val, imagenome_lookup, val_transform)
    test_ds = MSCXRTPairDataset(df_test, mscxrt_images_root, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    criterion_weight = compute_class_weights(df_train["label"].values).to(device) if class_weight else None
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)

    encoder, embed_dim = load_encoder(model_name, device)
    model = ConcatPairClassifier(encoder, embed_dim).to(device)

    # Warmup: head only
    for p in model.backbone_parameters():
        p.requires_grad_(False)
    opt_warmup = AdamW(model.head_parameters(), lr=head_lr, weight_decay=weight_decay)
    for _ in range(warmup_epochs):
        model.train()
        for img_prior, img_curr, labels in train_loader:
            loss = criterion(model(img_prior.to(device), img_curr.to(device)), labels.to(device))
            opt_warmup.zero_grad(); loss.backward(); opt_warmup.step()

    # Full fine-tuning with linear LR schedule
    for p in model.backbone_parameters():
        p.requires_grad_(True)
    opt = AdamW([
        {"params": model.backbone_parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": model.head_parameters(), "lr": head_lr, "weight_decay": weight_decay},
    ])
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(opt, lr_lambda)

    best_val_acc, best_state, no_improve = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for img_prior, img_curr, labels in tqdm(train_loader, desc=f"ep{ep+1}", leave=False):
            loss = criterion(model(img_prior.to(device), img_curr.to(device)), labels.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
            total_loss += loss.item()

        val_acc, val_f1 = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if ckpt_dir:
                torch.save(best_state, Path(ckpt_dir) / f"{model_name}_{finding}_seed{seed}_best.pt")
        else:
            no_improve += 1

        logger.info("[%s %s seed=%d] ep=%d  loss=%.4f  val_acc=%.3f  (best=%.3f)", model_name, finding, seed, ep+1, total_loss/len(train_loader), val_acc, best_val_acc)
        if no_improve >= patience:
            logger.info("[%s %s seed=%d] Early stop at epoch %d", model_name, finding, seed, ep+1)
            break

    model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    logger.info("[%s %s seed=%d] Test — macro_acc=%.3f  macro_f1=%.3f", model_name, finding, seed, test_acc, test_f1)
    return {"model": model_name, "finding": finding, "seed": seed, "macro_acc": test_acc, "macro_f1": test_f1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g. 'ours')")
    parser.add_argument("--train_pairs", default="data/imagenome_pairs/pairs_train.csv")
    parser.add_argument("--val_pairs", default="data/imagenome_pairs/pairs_val.csv")
    parser.add_argument("--dicom_to_path_csv", default="data/imagenome_pairs/dicom_to_path.csv")
    parser.add_argument("--mscxrt_labels", default="data/raw/ms_cxr_t_labels.csv")
    parser.add_argument("--imagenome_images", default="/data/imagenome_images")
    parser.add_argument("--mscxrt_images", default="data/raw/images")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--finding", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ckpt_dir", default="checkpoints/imagenome_generic")
    args = parser.parse_args()

    import pandas as pd
    from data.dataset import load_labels, FINDINGS
    from scripts.train_biovil_t_imagenome import build_dicom_lookup

    seeds = [args.seed] if args.seed is not None else args.seeds
    findings = [args.finding] if args.finding else FINDINGS

    df_train_all = pd.read_csv(args.train_pairs)
    df_val_all = pd.read_csv(args.val_pairs)
    df_test_all = load_labels(args.mscxrt_labels, images_root=args.mscxrt_images)

    imagenome_lookup = build_dicom_lookup(Path(args.imagenome_images), args.dicom_to_path_csv)
    logger.info("Image lookup: %d entries", len(imagenome_lookup))

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    for finding in findings:
        df_tr = df_train_all[df_train_all["finding"] == finding].copy()
        df_va = df_val_all[df_val_all["finding"] == finding].copy()
        df_te = df_test_all[df_test_all["finding"] == finding].copy()
        df_tr = df_tr[df_tr["curr_dicom_id"].isin(imagenome_lookup) & df_tr["prior_dicom_id"].isin(imagenome_lookup)]
        df_va = df_va[df_va["curr_dicom_id"].isin(imagenome_lookup) & df_va["prior_dicom_id"].isin(imagenome_lookup)]
        if len(df_tr) == 0 or len(df_te) == 0:
            continue

        for seed in seeds:
            result = train_one_finding(df_tr, df_va, df_te, finding=finding, seed=seed, model_name=args.model, imagenome_lookup=imagenome_lookup, mscxrt_images_root=args.mscxrt_images, device=args.device, epochs=args.epochs, batch_size=args.batch_size, backbone_lr=args.backbone_lr, patience=args.patience, ckpt_dir=args.ckpt_dir)
            if result:
                all_results.append(result)

    if not all_results:
        return

    import pandas as pd
    df_res = pd.DataFrame(all_results)
    for finding in findings:
        sub = df_res[df_res["finding"] == finding]
        if len(sub):
            logger.info("  %-20s  macro_acc=%.3f±%.3f", finding, sub["macro_acc"].mean(), sub["macro_acc"].std())
    per_seed = df_res.groupby("seed")["macro_acc"].mean()
    logger.info("  %-20s  macro_acc=%.3f±%.3f", "AVERAGE", per_seed.mean(), per_seed.std())

    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path("results") / f"{args.model}_imagenome_{timestamp}.json"
    with open(out, "w") as f:
        json.dump({"model": args.model, "protocol": "imagenome_finetune", "timestamp": timestamp, "results": all_results}, f, indent=2)
    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
