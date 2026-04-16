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

    Head init uses PyTorch's default Kaiming-uniform for Linear layers — a
    near-zero std=0.01 init on the final layer combined with balanced classes
    and weighted CE produced a symmetry-locked saddle point (predictions
    uniform, gradient ≈0, loss pinned at log K). Kaiming default breaks the
    symmetry by giving varied initial logits across classes.
    """

    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int = 3, hidden: int = 256):
        super().__init__()
        self.encoder = encoder  # must implement forward(img) → (B, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def set_final_bias_to_log_priors(self, label_counts: np.ndarray) -> None:
        """Initialise final-layer bias to log class priors so the model starts
        at the training class distribution (the correct starting point before
        seeing features). Helps gradient signal when classes are balanced.
        """
        priors = label_counts / label_counts.sum()
        log_priors = np.log(np.clip(priors, 1e-6, None)).astype(np.float32)
        final = self.head[-1]
        with torch.no_grad():
            final.bias.copy_(torch.from_numpy(log_priors - log_priors.mean()))

    def forward(self, img_prior: torch.Tensor, img_curr: torch.Tensor) -> torch.Tensor:
        f_prior = self.encoder(img_prior)   # (B, D)
        f_curr = self.encoder(img_curr)     # (B, D)
        pair = torch.cat([f_prior, f_curr], dim=1)  # (B, 2D)
        return self.head(pair)

    def backbone_parameters(self):
        return self.encoder.parameters()

    def head_parameters(self):
        return self.head.parameters()


class CrossAttnPairClassifier(nn.Module):
    """Siamese encoder with a minimal attention head over POOLED pair features.

    Each image is encoded to a single D-dim vector (the normal pooled output).
    Those two vectors become a 2-token sequence with a learnable type/position
    embedding to distinguish prior vs current. A learnable CLS query attends
    over that sequence via a single MultiheadAttention, then a linear layer
    produces logits.

    This is "attention as a smart learned pooling over prior+current" — far
    simpler than a full transformer-decoder stack over spatial tokens (which
    class-collapsed on the small per-finding datasets).
    """

    def __init__(
        self, encoder: nn.Module, embed_dim: int,
        num_classes: int = 3, num_heads: int = 8,
    ):
        super().__init__()
        self.encoder = encoder   # forward(img) → (B, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        # 2 type embeddings: one for prior, one for current.
        self.type_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))
        nn.init.normal_(self.type_embed, std=0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, img_prior: torch.Tensor, img_curr: torch.Tensor) -> torch.Tensor:
        f_prior = self.encoder(img_prior)                # (B, D)
        f_curr = self.encoder(img_curr)                  # (B, D)
        kv = torch.stack([f_prior, f_curr], dim=1) + self.type_embed  # (B, 2, D)
        B = kv.shape[0]
        q = self.cls_token.expand(B, -1, -1)             # (B, 1, D)
        attn_out, _ = self.attn(query=q, key=kv, value=kv)  # (B, 1, D)
        return self.head(self.norm(attn_out[:, 0]))

    def set_final_bias_to_log_priors(self, label_counts: np.ndarray) -> None:
        priors = label_counts / label_counts.sum()
        log_priors = np.log(np.clip(priors, 1e-6, None)).astype(np.float32)
        with torch.no_grad():
            self.head.bias.copy_(torch.from_numpy(log_priors - log_priors.mean()))

    def backbone_parameters(self):
        return self.encoder.parameters()

    def head_parameters(self):
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        return (p for p in self.parameters() if id(p) not in encoder_ids)


def load_encoder(model_name: str, device: str):
    """Load a PyTorch encoder by name. Returns (encoder_module, embed_dim).

    The returned module must accept a (B, 3, H, W) ImageNet-normalized tensor
    and return a (B, embed_dim) global-pooled embedding.
    """
    if model_name == "imagenet_densenet121":
        from torchvision.models import densenet121, DenseNet121_Weights
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()  # forward now returns (B, 1024)
        return m, 1024

    if model_name == "imagenet_resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()  # forward now returns (B, 2048)
        return m, 2048

    if model_name == "ours":
        raise NotImplementedError(
            "Our foundation model is not yet integrated. "
            "Add it to models/ours.py and update this function."
        )

    raise ValueError(
        f"Unknown model '{model_name}'. "
        f"Supported: imagenet_densenet121, imagenet_resnet50, ours. "
        f"For BioViL-T use train_biovil_t_imagenome.py. "
        f"For Google CXR use extract_google_cxr_imagenome_features.py + train_google_cxr_imagenome.py."
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
    device, epochs=30, warmup_epochs=1, batch_size=32,
    grad_accum_steps=4, use_amp=True, head_type="concat",
    backbone_lr=1e-5, head_lr=1e-3, weight_decay=1e-4,
    patience=10, num_workers=8, class_weight=True, ckpt_dir=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    from torchvision import transforms

    # ImageNet-pretrained models expect 3-channel RGB + ImageNet-normalised input.
    # health_multimodal.load_image returns PIL 'L' (grayscale); .convert('RGB')
    # replicates the single channel 3× which is the standard CXR-on-ImageNet
    # fine-tuning convention (CheXNet, TorchXRayVision, etc.).
    to_rgb = transforms.Lambda(lambda img: img.convert("RGB"))
    train_transform = transforms.Compose([
        to_rgb,
        transforms.Resize(512),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=30, shear=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])
    val_transform = transforms.Compose([
        to_rgb,
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImaGenomePairDataset(df_train, imagenome_lookup, train_transform)
    val_ds = ImaGenomePairDataset(df_val, imagenome_lookup, val_transform)
    test_ds = MSCXRTPairDataset(df_test, mscxrt_images_root, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    criterion_weight = compute_class_weights(df_train["label"].values).to(device) if class_weight else None
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)

    encoder, embed_dim = load_encoder(model_name, device)
    if head_type == "concat":
        model = ConcatPairClassifier(encoder, embed_dim)
    elif head_type == "attention":
        model = CrossAttnPairClassifier(encoder, embed_dim)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")
    label_counts = np.bincount(df_train["label"].values, minlength=3)
    model.set_final_bias_to_log_priors(label_counts)
    model = model.to(device)
    # bf16 avoids fp16 overflow issues common with DenseNet-style concat
    # architectures and doesn't need GradScaler (fp32-range exponent).
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    use_scaler = False  # bf16 / fp32 don't need loss scaling
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Warmup: head only
    for p in model.backbone_parameters():
        p.requires_grad_(False)
    opt_warmup = AdamW(model.head_parameters(), lr=head_lr, weight_decay=weight_decay)
    for _ in range(warmup_epochs):
        model.train()
        for img_prior, img_curr, labels in train_loader:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = criterion(model(img_prior.to(device), img_curr.to(device)), labels.to(device))
            opt_warmup.zero_grad()
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(opt_warmup)
                scaler.update()
            else:
                loss.backward()
                opt_warmup.step()

    # Full fine-tuning with linear LR schedule + gradient accumulation
    for p in model.backbone_parameters():
        p.requires_grad_(True)
    opt = AdamW([
        {"params": model.backbone_parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": model.head_parameters(), "lr": head_lr, "weight_decay": weight_decay},
    ])
    steps_per_epoch = max(1, len(train_loader) // grad_accum_steps)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

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
        opt.zero_grad()
        for step, (img_prior, img_curr, labels) in enumerate(
            tqdm(train_loader, desc=f"ep{ep+1}", leave=False)
        ):
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(img_prior.to(device), img_curr.to(device))
                loss = criterion(logits, labels.to(device)) / grad_accum_steps
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item() * grad_accum_steps
            if (step + 1) % grad_accum_steps == 0:
                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                scheduler.step()
                opt.zero_grad()

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
    logger.info(
        "[%s %s seed=%d] MS-CXR-T test — macro_acc=%.3f  macro_f1=%.3f  (best_val=%.3f)",
        model_name, finding, seed, test_acc, test_f1, best_val_acc,
    )
    return {
        "model": model_name, "finding": finding, "seed": seed,
        "macro_acc": test_acc, "macro_f1": test_f1,
        "best_val_macro_acc": best_val_acc,
    }


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
    parser.add_argument("--imagenome_images", default="/data/mimic-cxr-jpg")
    parser.add_argument("--mscxrt_images", default="data/raw/images")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--finding", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-step batch size. Effective batch = batch_size * grad_accum_steps.")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Paper effective batch 128 = 32 × 4.")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--head_type", choices=["concat", "attention"], default="concat",
                        help="concat: global-pooled feats → MLP (default). "
                             "attention: cross-attention over spatial tokens (DenseNet only).")
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
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
        logger.info(
            "Finding: %s — train=%d, val=%d, test(MS-CXR-T)=%d",
            finding, len(df_tr), len(df_va), len(df_te),
        )

        for seed in seeds:
            result = train_one_finding(
                df_tr, df_va, df_te,
                finding=finding, seed=seed, model_name=args.model,
                imagenome_lookup=imagenome_lookup,
                mscxrt_images_root=args.mscxrt_images,
                device=args.device,
                epochs=args.epochs, warmup_epochs=args.warmup_epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                use_amp=not args.no_amp,
                head_type=args.head_type,
                backbone_lr=args.backbone_lr, head_lr=args.head_lr,
                patience=args.patience, num_workers=args.num_workers,
                ckpt_dir=args.ckpt_dir,
            )
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
    tag = args.head_type if args.head_type != "concat" else ""
    suffix = f"_{tag}" if tag else ""
    out = Path("results") / f"{args.model}{suffix}_imagenome_{timestamp}.json"
    with open(out, "w") as f:
        json.dump({
            "model": args.model, "head_type": args.head_type,
            "protocol": "imagenome_finetune", "timestamp": timestamp,
            "args": vars(args), "results": all_results,
        }, f, indent=2)
    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
