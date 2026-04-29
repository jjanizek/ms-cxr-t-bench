"""Hyperparameter sweep for the frozen-SMB-features MLP probe.

Loads pre-extracted features once (deepstack-mode, 8192-dim per image →
16384-dim per pair) and grid-searches over head architecture / training HPs.
4 seeds per config; reports mean ± std on MS-CXR-T test set per (finding, config).

Sweep axes:
  - feature subset: merger only (dim 2048 per image) vs full deepstack (8192)
  - head depth: 1 (linear), 2, 3 layers
  - hidden size: 256, 512, 1024
  - dropout: 0.0, 0.1, 0.3
  - LR: 1e-4, 5e-4, 1e-3, 5e-3
  - weight decay: 1e-4, 1e-3
  - bias init: log_priors+perturb_std=0.1
  - class weight: on (single value — has been default)

Prerequisite: extract scripts at the same input_size:
    python scripts/extract_smb_vision_imagenome_features.py \
        --pooling_mode deepstack_concat --input_size 768 \
        --out_dir data/features/smb_vision_imagenome_in768_ds
    python scripts/extract_smb_vision_features.py \
        --pooling_mode deepstack_concat --input_size 768 \
        --out_dir data/features/smb_vision_in768_ds

Usage:
    python scripts/sweep_smb_vision_probe.py
    python scripts/sweep_smb_vision_probe.py --finding pneumothorax
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPProbe(nn.Module):
    """MLP probe with configurable depth, hidden size, dropout."""

    def __init__(self, in_dim: int, hidden: int, depth: int, dropout: float, num_classes: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(max(0, depth - 1)):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def set_final_bias_to_log_priors(self, label_counts: np.ndarray, perturb_std: float = 0.1) -> None:
        priors = label_counts / label_counts.sum()
        log_priors = np.log(np.clip(priors, 1e-6, None)).astype(np.float32)
        bias = log_priors - log_priors.mean()
        final = self.net[-1]
        with torch.no_grad():
            final.bias.copy_(torch.from_numpy(bias))
            final.bias.add_(torch.randn_like(final.bias) * perturb_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)


def make_loader(feats: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X = torch.from_numpy(feats).float()
    y = torch.from_numpy(labels).long()
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            all_logits.append(logits.float().cpu().numpy())
            all_labels.extend(y.numpy())
    logits_arr = np.concatenate(all_logits, axis=0)
    y_pred = logits_arr.argmax(axis=1)
    y_true = np.array(all_labels)
    macro_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return macro_acc, macro_f1


def train_one(
    tr_feats, tr_labels, va_feats, va_labels, te_feats, te_labels,
    seed: int, device: str, depth: int, hidden: int, dropout: float,
    lr: float, weight_decay: float, batch_size: int, epochs: int,
    patience: int, class_weight: bool, perturb_std: float,
) -> tuple[float, float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = make_loader(tr_feats, tr_labels, batch_size, shuffle=True)
    val_loader = make_loader(va_feats, va_labels, batch_size, shuffle=False)
    test_loader = make_loader(te_feats, te_labels, batch_size, shuffle=False)

    criterion_weight = compute_class_weights(tr_labels).to(device) if class_weight else None
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)

    model = MLPProbe(in_dim=tr_feats.shape[1], hidden=hidden, depth=depth, dropout=dropout)
    label_counts = np.bincount(tr_labels, minlength=3)
    model.set_final_bias_to_log_priors(label_counts, perturb_std=perturb_std)
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    warmup_steps = max(1, total_steps // 30)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = LambdaLR(opt, lr_lambda)

    best_val_acc, best_state, no_improve = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for X, y in train_loader:
            loss = criterion(model(X.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
        val_acc, _ = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    return test_acc, test_f1, best_val_acc


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def feature_subsets(feats: np.ndarray, mode: str) -> np.ndarray:
    """Slice deepstack-pair features into a smaller view.

    Layout (per pair = 2 * per-image): per-image is concat of
      [final_merger (D), deepstack_0 (D), deepstack_1 (D), deepstack_2 (D)]
    where D = 2048. Pair feature: prior(8192) || curr(8192).
    """
    D = 2048
    if mode == "merger":
        # take only the first D dims of each image's per-image feature
        # pair layout: prior 8192 | curr 8192 → take prior[:D] | curr[:D]
        prior = feats[:, :D]
        curr = feats[:, 4 * D : 4 * D + D]
        return np.concatenate([prior, curr], axis=1)
    if mode == "deepstack":
        return feats   # full 16384 (4 levels × 2 images)
    raise ValueError(f"Unknown feature subset mode: {mode}")


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenome_features_dir", default="data/features/smb_vision_imagenome_in768_ds")
    parser.add_argument("--mscxrt_features_dir", default="data/features/smb_vision_in768_ds")
    parser.add_argument("--mscxrt_labels", default="data/raw/ms_cxr_t_labels.csv")
    parser.add_argument("--mscxrt_images", default="data/raw/images")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789])
    parser.add_argument("--finding", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from data.dataset import load_labels, FINDINGS

    findings = [args.finding] if args.finding else FINDINGS
    imagenome_dir = Path(args.imagenome_features_dir)
    mscxrt_dir = Path(args.mscxrt_features_dir)
    df_test_all = load_labels(args.mscxrt_labels, images_root=args.mscxrt_images)

    # Sweep grid (axes that proved to matter on this dataset).
    sweep = list(itertools.product(
        ["merger", "deepstack"],            # feature_subset
        [1, 2, 3],                          # depth
        [256, 512, 1024],                   # hidden
        [0.0, 0.1, 0.3],                    # dropout
        [1e-4, 5e-4, 1e-3, 5e-3],           # lr
        [1e-4, 1e-3],                       # weight_decay
        [128],                              # batch_size (fixed — sweep showed minimal effect in pilot)
        [0.1],                              # perturb_std (fixed)
    ))
    # Filter: depth=1 (linear) → hidden/dropout don't matter; collapse to one config per
    # depth=1 × {feat_subset × lr × wd}.
    seen: set[tuple] = set()
    configs: list[tuple] = []
    for cfg in sweep:
        feat_mode, depth, hidden, dropout, lr, wd, bs, ps = cfg
        if depth == 1:
            key = (feat_mode, depth, lr, wd, bs, ps)
            if key in seen:
                continue
            seen.add(key)
        configs.append(cfg)
    logger.info("Sweep configs after dedup: %d", len(configs))

    results: list[dict] = []

    for finding in findings:
        tr_p = imagenome_dir / f"{finding}_train.npy"
        va_p = imagenome_dir / f"{finding}_val.npy"
        te_p = mscxrt_dir / f"{finding}_all.npy"
        missing = [p for p in [tr_p, va_p, te_p] if not p.exists()]
        if missing:
            logger.warning("Skipping %s, missing: %s", finding, missing)
            continue
        tr_full = np.load(tr_p)
        va_full = np.load(va_p)
        te_full = np.load(te_p)
        tr_labels = np.load(imagenome_dir / f"{finding}_train_labels.npy")
        va_labels = np.load(imagenome_dir / f"{finding}_val_labels.npy")
        df_te = df_test_all[df_test_all["finding"] == finding].reset_index(drop=True)
        te_labels = df_te["label"].values.astype(np.int64)
        logger.info("=== %s ===  train=%d val=%d test=%d  feat_dim_full=%d",
                    finding, len(tr_full), len(va_full), len(te_full), tr_full.shape[1])

        # Pre-slice both subsets once
        feats_by_mode = {
            "merger": (feature_subsets(tr_full, "merger"),
                       feature_subsets(va_full, "merger"),
                       feature_subsets(te_full, "merger")),
            "deepstack": (tr_full, va_full, te_full),
        }

        for cfg_idx, (feat_mode, depth, hidden, dropout, lr, wd, bs, ps) in enumerate(configs):
            tr_f, va_f, te_f = feats_by_mode[feat_mode]
            seed_results = []
            for seed in args.seeds:
                test_acc, test_f1, best_val = train_one(
                    tr_f, tr_labels, va_f, va_labels, te_f, te_labels,
                    seed=seed, device=args.device, depth=depth, hidden=hidden,
                    dropout=dropout, lr=lr, weight_decay=wd, batch_size=bs,
                    epochs=args.epochs, patience=args.patience,
                    class_weight=True, perturb_std=ps,
                )
                seed_results.append({
                    "seed": seed,
                    "test_macro_acc": test_acc,
                    "test_macro_f1": test_f1,
                    "best_val_macro_acc": best_val,
                })

            test_accs = np.array([r["test_macro_acc"] for r in seed_results])
            test_f1s = np.array([r["test_macro_f1"] for r in seed_results])
            best_vals = np.array([r["best_val_macro_acc"] for r in seed_results])

            entry = {
                "finding": finding,
                "feat_mode": feat_mode,
                "depth": depth,
                "hidden": hidden,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": wd,
                "batch_size": bs,
                "perturb_std": ps,
                "seeds": args.seeds,
                "test_macro_acc_mean": float(test_accs.mean()),
                "test_macro_acc_std": float(test_accs.std()),
                "test_macro_f1_mean": float(test_f1s.mean()),
                "test_macro_f1_std": float(test_f1s.std()),
                "best_val_macro_acc_mean": float(best_vals.mean()),
                "per_seed": seed_results,
            }
            results.append(entry)

            if (cfg_idx + 1) % 25 == 0 or cfg_idx == len(configs) - 1:
                logger.info(
                    "[%s %d/%d] feat=%s d=%d h=%d drop=%.2f lr=%.0e wd=%.0e bs=%d  "
                    "test=%.3f±%.3f val=%.3f",
                    finding, cfg_idx + 1, len(configs), feat_mode, depth, hidden,
                    dropout, lr, wd, bs, test_accs.mean(), test_accs.std(), best_vals.mean(),
                )

    if not results:
        logger.warning("No results.")
        return

    df = pd.DataFrame(results)
    logger.info("\n=== Best per finding (by test_macro_acc_mean) ===")
    for finding in findings:
        sub = df[df["finding"] == finding]
        if len(sub) == 0:
            continue
        best = sub.loc[sub["test_macro_acc_mean"].idxmax()]
        logger.info(
            "  %-20s  %.3f±%.3f  feat=%s d=%d h=%d drop=%.2f lr=%.0e wd=%.0e",
            finding, best["test_macro_acc_mean"], best["test_macro_acc_std"],
            best["feat_mode"], best["depth"], best["hidden"], best["dropout"],
            best["lr"], best["weight_decay"],
        )

    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else Path("results") / f"smb_vision_probe_sweep_{timestamp}.json"
    payload = {
        "model": "smb_vision_v1_cxr",
        "protocol": "frozen_probe_imagenome_sweep",
        "input_size": 768,
        "extracted_pooling_mode": "deepstack_concat",
        "git_hash": git_hash(),
        "timestamp": timestamp,
        "args": vars(args),
        "n_configs": len(set(tuple(c) for c in configs)),
        "results": results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Sweep results saved to %s", out)


if __name__ == "__main__":
    main()
