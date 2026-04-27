"""Train a frozen-feature MLP probe for SMB Vision v1 CXR on ImaGenome, test on MS-CXR-T.

The end-to-end fine-tune of this 600M model collapsed for 2/5 findings (saddle
points + over-parameterised backbone destroying pretrained features). A frozen
encoder + MLP probe — the same protocol we use for Google CXR — preserves the
pretrained representations and limits supervised work to the small head.

Architecture:
  [prior_feat ; curr_feat]  (2*2048 = 4096-dim)  →  MLP(4096 → 512 → 3)

Prerequisites:
    python scripts/extract_smb_vision_imagenome_features.py
    python scripts/extract_smb_vision_features.py

Usage:
    python scripts/train_smb_vision_imagenome.py
    python scripts/train_smb_vision_imagenome.py --finding edema --seed 42
"""
import argparse
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class MLPProbe(nn.Module):
    """2-layer MLP for frozen-feature pair classification.

    Init choice matters: a near-zero std=0.01 weight init plus zero bias
    combined with class-balanced cross-entropy puts the optimiser at a
    symmetry-locked saddle (logits ≈ 0 → softmax = (1/3,1/3,1/3) → balanced
    CE gradient ≈ 0). PyTorch's default Kaiming-uniform init breaks the
    symmetry by giving each class a different initial logit. The final-layer
    bias is set to log class priors so we start at the training distribution.
    """

    def __init__(self, in_dim: int, hidden: int = 512, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
        # Default Kaiming-uniform init; do not zero the small-std overrides.

    def set_final_bias_to_log_priors(self, label_counts: np.ndarray) -> None:
        priors = label_counts / label_counts.sum()
        log_priors = np.log(np.clip(priors, 1e-6, None)).astype(np.float32)
        final = self.net[-1]
        with torch.no_grad():
            final.bias.copy_(torch.from_numpy(log_priors - log_priors.mean()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)


def make_loader(feats: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X = torch.from_numpy(feats).float()
    y = torch.from_numpy(labels).long()
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def evaluate(model: nn.Module, loader: DataLoader, device: str, return_logits: bool = False):
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
    if return_logits:
        return macro_acc, macro_f1, logits_arr, y_true
    return macro_acc, macro_f1


def train_one_finding(
    tr_feats, tr_labels, va_feats, va_labels, te_feats, te_labels,
    finding: str, seed: int, device: str,
    epochs: int = 30, batch_size: int = 128,
    lr: float = 1e-3, weight_decay: float = 1e-4,
    patience: int = 10, class_weight: bool = True,
    ckpt_dir: Path | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = tr_feats.shape[1]

    train_loader = make_loader(tr_feats, tr_labels, batch_size, shuffle=True)
    val_loader = make_loader(va_feats, va_labels, batch_size, shuffle=False)
    test_loader = make_loader(te_feats, te_labels, batch_size, shuffle=False)

    criterion_weight = compute_class_weights(tr_labels).to(device) if class_weight else None
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)

    model = MLPProbe(in_dim=in_dim)
    label_counts = np.bincount(tr_labels, minlength=3)
    model.set_final_bias_to_log_priors(label_counts)
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, total_steps // 30)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(opt, lr_lambda)

    best_val_acc, best_state, no_improve = -1.0, None, 0

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"{finding}/seed{seed}/ep{ep+1}", leave=False):
            loss = criterion(model(X.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            total_loss += loss.item()

        val_acc, val_f1 = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if ckpt_dir:
                torch.save(best_state, ckpt_dir / f"smb_vision_{finding}_seed{seed}_best.pt")
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

    model.load_state_dict(best_state)
    test_acc, test_f1, test_logits, test_y_true = evaluate(
        model, test_loader, device, return_logits=True,
    )
    logger.info(
        "[%s seed=%d] MS-CXR-T test — macro_acc=%.3f  macro_f1=%.3f  (best_val=%.3f)",
        finding, seed, test_acc, test_f1, best_val_acc,
    )

    preds_dir = Path("results/predictions/smb_vision_imagenome")
    preds_dir.mkdir(parents=True, exist_ok=True)
    label_names = ["improving", "stable", "worsening"]
    preds_df = pd.DataFrame({
        "finding": finding,
        "gt_label": test_y_true,
        "ground_truth": [label_names[int(y)] for y in test_y_true],
        "pred_label": test_logits.argmax(axis=1),
        "predicted": [label_names[int(p)] for p in test_logits.argmax(axis=1)],
        "logit_improving": test_logits[:, 0],
        "logit_stable": test_logits[:, 1],
        "logit_worsening": test_logits[:, 2],
    })
    preds_path = preds_dir / f"{finding}_seed{seed}_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    logger.info("  saved per-sample predictions → %s", preds_path)

    return {
        "model": "smb_vision_imagenome",
        "finding": finding,
        "seed": seed,
        "macro_acc": test_acc,
        "macro_f1": test_f1,
        "best_val_macro_acc": best_val_acc,
        "predictions_csv": str(preds_path),
    }


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenome_features_dir", default="data/features/smb_vision_imagenome")
    parser.add_argument("--mscxrt_features_dir", default="data/features/smb_vision")
    parser.add_argument("--mscxrt_labels", default="data/raw/ms_cxr_t_labels.csv")
    parser.add_argument("--mscxrt_images", default="data/raw/images")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--finding", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no_class_weight", action="store_true")
    parser.add_argument("--ckpt_dir", default="checkpoints/smb_vision_imagenome")
    args = parser.parse_args()

    from data.dataset import load_labels, FINDINGS

    seeds = [args.seed] if args.seed is not None else args.seeds
    findings = [args.finding] if args.finding else FINDINGS

    imagenome_dir = Path(args.imagenome_features_dir)
    mscxrt_dir = Path(args.mscxrt_features_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    df_test_all = load_labels(args.mscxrt_labels, images_root=args.mscxrt_images)
    logger.info("MS-CXR-T: %d (finding, pair) rows", len(df_test_all))

    all_results = []

    for finding in findings:
        tr_feats_path = imagenome_dir / f"{finding}_train.npy"
        tr_labels_path = imagenome_dir / f"{finding}_train_labels.npy"
        va_feats_path = imagenome_dir / f"{finding}_val.npy"
        va_labels_path = imagenome_dir / f"{finding}_val_labels.npy"
        te_feats_path = mscxrt_dir / f"{finding}_all.npy"

        missing = [p for p in [tr_feats_path, tr_labels_path, te_feats_path] if not p.exists()]
        if missing:
            logger.warning(
                "Missing feature files for %s — skipping:\n  %s\n"
                "Run extract_smb_vision_imagenome_features.py / extract_smb_vision_features.py first.",
                finding,
                "\n  ".join(str(p) for p in missing),
            )
            continue

        tr_feats = np.load(tr_feats_path)
        tr_labels = np.load(tr_labels_path)

        if va_feats_path.exists() and va_labels_path.exists():
            va_feats = np.load(va_feats_path)
            va_labels = np.load(va_labels_path)
        else:
            logger.warning(
                "No val features for %s — using train as val (early stopping disabled)", finding
            )
            va_feats = tr_feats
            va_labels = tr_labels

        te_feats = np.load(te_feats_path)
        df_te = df_test_all[df_test_all["finding"] == finding].reset_index(drop=True)
        if len(df_te) != len(te_feats):
            raise ValueError(
                f"{finding}: MS-CXR-T labels has {len(df_te)} rows but "
                f"feature cache has {len(te_feats)} rows. Re-run extract_smb_vision_features.py."
            )
        te_labels = df_te["label"].values.astype(np.int64)

        logger.info(
            "Finding: %s — train=%d, val=%d, test(MS-CXR-T)=%d, feat_dim=%d",
            finding, len(tr_feats), len(va_feats), len(te_feats), tr_feats.shape[1],
        )

        for seed in seeds:
            result = train_one_finding(
                tr_feats, tr_labels, va_feats, va_labels, te_feats, te_labels,
                finding=finding, seed=seed, device=args.device,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                class_weight=not args.no_class_weight,
                ckpt_dir=ckpt_dir,
            )
            if result:
                all_results.append(result)

    if not all_results:
        logger.warning("No results produced.")
        return

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
    logger.info("  %-20s  macro_acc=%.3f±%.3f", "AVERAGE", per_seed.mean(), per_seed.std())

    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path("results") / f"smb_vision_imagenome_{timestamp}.json"
    payload = {
        "model": "smb_vision_imagenome",
        "protocol": "frozen_probe_imagenome",
        "description": (
            "Frozen SMB Vision v1 CXR features (merger-pooled 2048-dim per image), "
            "MLP probe trained on ImaGenome silver labels, "
            "tested on full MS-CXR-T (comparable to BioViL-T Table 2 but frozen encoder)"
        ),
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
