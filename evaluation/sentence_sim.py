"""Temporal sentence similarity evaluation.

Cosine similarity + threshold tuning via 10-fold CV.
Subsets: RadGraph (117 pairs), Swaps (244 pairs).
Metrics: Accuracy, ROC-AUC (binary: paraphrase vs. contradiction).
"""
import logging
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N, D) arrays."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a_norm * b_norm).sum(axis=1)


def threshold_search(sims: np.ndarray, labels: np.ndarray) -> float:
    """Find threshold maximising accuracy on provided split."""
    best_t, best_acc = 0.0, 0.0
    for t in np.linspace(sims.min(), sims.max(), 200):
        preds = (sims >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t


def run_sentence_similarity(
    text_feats_1: np.ndarray,
    text_feats_2: np.ndarray,
    labels: np.ndarray,
    subset: str,
    n_splits: int = 10,
    seed: int = 42,
) -> dict:
    """10-fold CV threshold search for sentence similarity.

    Args:
        text_feats_1: (N, D) embeddings for the first sentence.
        text_feats_2: (N, D) embeddings for the second sentence.
        labels: (N,) binary — 1 = paraphrase, 0 = contradiction.
        subset: "RadGraph" or "Swaps".

    Returns:
        dict with accuracy and roc_auc (averaged over folds).
    """
    sims = cosine_sim(text_feats_1, text_feats_2)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_accs, fold_aucs = [], []

    for train_idx, test_idx in skf.split(sims, labels):
        t = threshold_search(sims[train_idx], labels[train_idx])
        preds = (sims[test_idx] >= t).astype(int)
        fold_accs.append(accuracy_score(labels[test_idx], preds))
        try:
            fold_aucs.append(roc_auc_score(labels[test_idx], sims[test_idx]))
        except ValueError:
            pass  # degenerate fold

    return {
        "subset": subset,
        "accuracy": float(np.mean(fold_accs)),
        "roc_auc": float(np.mean(fold_aucs)) if fold_aucs else None,
        "n_samples": len(labels),
    }
