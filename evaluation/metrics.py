"""Shared metric computation for MS-CXR-T benchmarking."""
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def macro_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro (balanced) accuracy: mean per-class recall."""
    classes = np.unique(y_true)
    per_class = [
        accuracy_score(y_true[y_true == c], y_pred[y_true == c])
        for c in classes
    ]
    return float(np.mean(per_class))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def macro_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_classes: int = 3,
) -> Optional[float]:
    """One-vs-rest macro AUROC.  Returns None if a class is missing from y_true."""
    try:
        return float(
            roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        )
    except ValueError:
        return None


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> dict:
    """Compute macro accuracy, F1, and (optionally) AUROC.

    Args:
        y_true:  (N,) integer class labels.
        y_pred:  (N,) predicted class labels.
        y_score: (N, C) softmax probabilities, or None.

    Returns:
        dict with keys: macro_acc, macro_f1, macro_auroc (may be None).
    """
    return {
        "macro_acc": macro_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_auroc": macro_auroc(y_true, y_score) if y_score is not None else None,
    }
