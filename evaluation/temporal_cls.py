"""Temporal image classification evaluation (Protocols A, B, C).

Protocol A: frozen-encoder probe — train/val/test, concatenated features.
  probe_type options: "linear" | "mlp" | "svm" | "logreg"
Protocol B: SVM — 5-fold CV on concatenated features.
Protocol C: zero-shot / cosine similarity (model-specific).
"""
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import compute_all

logger = logging.getLogger(__name__)

FINDINGS = [
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]


# ---------------------------------------------------------------------------
# Neural probes (linear + MLP)
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class MLPProbe(nn.Module):
    """MLP probe with configurable hidden layers, BatchNorm, and Dropout."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 3,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256]
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _train_nn_probe(
    probe: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 64,
    device: str = "cpu",
) -> nn.Module:
    """Generic training loop for any nn.Module probe, with val-acc early stopping."""
    probe = probe.to(device)
    opt = Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.from_numpy(X_train).float().to(device)
    y_tr = torch.from_numpy(y_train).long().to(device)
    X_v = torch.from_numpy(X_val).float().to(device)

    best_val_acc = -1.0
    best_state = None

    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i : i + batch_size]
            logits = probe(X_tr[idx])
            loss = criterion(logits, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            val_preds = probe(X_v).argmax(1).cpu().numpy()
        val_acc = (val_preds == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    return probe


def _eval_nn_probe(probe: nn.Module, X_test: np.ndarray, y_test: np.ndarray, device: str = "cpu") -> dict:
    probe.eval()
    X = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad():
        logits = probe(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
    return compute_all(y_test, preds, probs)


# ---------------------------------------------------------------------------
# sklearn probes (SVM, LogReg) — use train+val combined, no epochs needed
# ---------------------------------------------------------------------------

def _train_eval_sklearn(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Fit on train+val (sklearn doesn't need a separate val set), eval on test."""
    scaler = StandardScaler()
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    X_all = scaler.fit_transform(X_all)
    X_test_s = scaler.transform(X_test)
    clf.fit(X_all, y_all)
    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s) if hasattr(clf, "predict_proba") else None
    return compute_all(y_test, preds, probs)


# ---------------------------------------------------------------------------
# Protocol A dispatcher
# ---------------------------------------------------------------------------

def run_protocol_a(
    features: dict,  # {"train": (X, y), "val": (X, y), "test": (X, y)}
    finding: str,
    seed: int,
    probe_type: str = "linear",
    **probe_kwargs,
) -> dict:
    """Run Protocol A for a single finding and seed.

    probe_type: "linear" | "mlp" | "svm" | "logreg"
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    (X_train, y_train) = features["train"]
    (X_val, y_val) = features["val"]
    (X_test, y_test) = features["test"]
    in_dim = X_train.shape[1]
    device = probe_kwargs.get("device", "cpu")

    if probe_type == "linear":
        probe = LinearProbe(in_dim)
        probe = _train_nn_probe(probe, X_train, y_train, X_val, y_val, **{
            k: v for k, v in probe_kwargs.items() if k != "device"
        }, device=device)
        metrics = _eval_nn_probe(probe, X_test, y_test, device=device)

    elif probe_type == "mlp":
        hidden_dims = probe_kwargs.pop("hidden_dims", [256])
        dropout = probe_kwargs.pop("dropout", 0.3)
        probe = MLPProbe(in_dim, hidden_dims=hidden_dims, dropout=dropout)
        probe = _train_nn_probe(probe, X_train, y_train, X_val, y_val, **{
            k: v for k, v in probe_kwargs.items() if k != "device"
        }, device=device)
        metrics = _eval_nn_probe(probe, X_test, y_test, device=device)

    elif probe_type == "svm":
        C = probe_kwargs.get("C", 1.0)
        clf = SVC(kernel="rbf", C=C, probability=True, random_state=seed)
        metrics = _train_eval_sklearn(clf, X_train, y_train, X_val, y_val, X_test, y_test)

    elif probe_type == "logreg":
        C = probe_kwargs.get("C", 1.0)
        clf = LogisticRegression(C=C, max_iter=1000, random_state=seed, multi_class="multinomial")
        metrics = _train_eval_sklearn(clf, X_train, y_train, X_val, y_val, X_test, y_test)

    else:
        raise ValueError(f"Unknown probe_type: {probe_type!r}. Choose: linear, mlp, svm, logreg")

    return {"finding": finding, "seed": seed, "probe_type": probe_type, **metrics}


# ---------------------------------------------------------------------------
# Protocol B: SVM 5-fold CV
# ---------------------------------------------------------------------------

def run_protocol_b(
    X: np.ndarray,
    y: np.ndarray,
    finding: str,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Protocol B: SVM with 5-fold stratified CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        svm = SVC(kernel="rbf", probability=True, random_state=seed)
        svm.fit(X[train_idx], y[train_idx])
        y_pred = svm.predict(X[test_idx])
        y_score = svm.predict_proba(X[test_idx])
        fold_metrics.append(compute_all(y[test_idx], y_pred, y_score))

    # Average across folds
    avg = {}
    for k in fold_metrics[0]:
        vals = [m[k] for m in fold_metrics if m[k] is not None]
        avg[k] = float(np.mean(vals)) if vals else None
    return {"finding": finding, "protocol": "B", **avg}
