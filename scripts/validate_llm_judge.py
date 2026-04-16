"""Validate LLM-as-judge extraction against human labels on a subset.

Two-phase workflow:
  1. EXPORT: sample ~100 MAIRA-2 reports, create a CSV for human annotation.
     Human reads the report text, writes their own label in the 'human_label' column.
  2. ANALYZE: load the annotated CSV, compute agreement stats, generate plots.

Outputs:
  - Annotation CSV (phase 1): data/maira2_judge_validation/annotation_sheet.csv
  - Agreement report + publication-ready confusion matrix plot (phase 2)

Usage:
    # Phase 1: generate annotation sheet
    python scripts/validate_llm_judge.py export \
        --predictions results/maira2_judge/predictions_*.csv \
        --n_samples 100

    # Phase 2: after human fills in human_label column
    python scripts/validate_llm_judge.py analyze \
        --annotated data/maira2_judge_validation/annotation_sheet.csv
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LABEL_MAP = {"improving": 0, "stable": 1, "worsening": 2}
LABEL_NAMES = ["improving", "stable", "worsening"]


def cmd_export(args):
    """Sample N reports and create an annotation CSV."""
    df = pd.read_csv(args.predictions)
    logger.info("Loaded %d predictions", len(df))

    n = min(args.n_samples, len(df))
    # Stratified sample: proportional to finding
    frames = []
    for finding, g in df.groupby("finding"):
        k = max(1, round(n * len(g) / len(df)))
        frames.append(g.sample(n=k, random_state=42))
    sampled = pd.concat(frames, ignore_index=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create annotation sheet
    sheet = sampled[["dicom_id", "previous_dicom_id", "finding",
                     "maira2_report", "predicted"]].copy()
    sheet["human_label"] = ""  # blank column for annotator
    sheet["notes"] = ""

    out_path = out_dir / "annotation_sheet.csv"
    sheet.to_csv(out_path, index=False)
    logger.info(
        "Annotation sheet saved to %s (%d samples, stratified by finding)",
        out_path, len(sheet),
    )
    logger.info("Columns: dicom_id, previous_dicom_id, finding, maira2_report, "
                "predicted (LLM judge), human_label (FILL THIS IN), notes")
    logger.info(
        "Finding distribution:\n%s",
        sheet["finding"].value_counts().to_string(),
    )


def cmd_analyze(args):
    """Compare human vs LLM labels, compute agreement, generate plots."""
    df = pd.read_csv(args.annotated)
    df = df[df["human_label"].notna() & (df["human_label"] != "")].copy()
    df["human_label"] = df["human_label"].str.lower().str.strip()
    df["predicted"] = df["predicted"].str.lower().str.strip()

    valid = df[df["human_label"].isin(LABEL_MAP) & df["predicted"].isin(LABEL_MAP)].copy()
    if len(valid) < len(df):
        logger.warning(
            "Dropped %d rows with invalid labels. Remaining: %d",
            len(df) - len(valid), len(valid),
        )

    y_human = valid["human_label"].map(LABEL_MAP).values
    y_llm = valid["predicted"].map(LABEL_MAP).values
    n = len(valid)

    # Agreement metrics
    acc = float(np.mean(y_human == y_llm))
    bal_acc = float(balanced_accuracy_score(y_human, y_llm))
    kappa = float(cohen_kappa_score(y_human, y_llm))
    cm = confusion_matrix(y_human, y_llm, labels=[0, 1, 2])

    logger.info("\n=== LLM Judge Validation (n=%d) ===", n)
    logger.info("  Raw agreement:      %.1f%%", 100 * acc)
    logger.info("  Balanced accuracy:  %.3f", bal_acc)
    logger.info("  Cohen's kappa:      %.3f", kappa)
    logger.info("\nClassification report (human as ground truth):\n%s",
                classification_report(y_human, y_llm, target_names=LABEL_NAMES, zero_division=0))

    # Per-finding agreement
    logger.info("Per-finding agreement:")
    for finding in valid["finding"].unique():
        sub = valid[valid["finding"] == finding]
        yh = sub["human_label"].map(LABEL_MAP).values
        yl = sub["predicted"].map(LABEL_MAP).values
        f_acc = float(np.mean(yh == yl))
        logger.info("  %-20s  n=%d  agreement=%.1f%%", finding, len(sub), 100 * f_acc)

    # Save metrics
    out_dir = Path(args.annotated).parent
    metrics = {
        "n_samples": n,
        "raw_agreement": acc,
        "balanced_accuracy": bal_acc,
        "cohens_kappa": kappa,
        "confusion_matrix": cm.tolist(),
        "labels": LABEL_NAMES,
    }
    metrics_path = out_dir / "judge_validation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Generate plot
    plot_path = out_dir / "judge_validation_confusion.png"
    make_confusion_plot(cm, LABEL_NAMES, acc, kappa, n, plot_path)
    logger.info("Plot saved to %s", plot_path)


def make_confusion_plot(cm, labels, accuracy, kappa, n, out_path):
    """Publication-ready confusion matrix: human (rows) vs LLM judge (cols)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.2))

    # Normalize by row (human label) to show recall per class
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    # Annotate cells with count and percentage
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i, j]
            pct = 100 * cm_norm[i, j]
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=11, color=color, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l.capitalize() for l in labels], fontsize=10)
    ax.set_yticklabels([l.capitalize() for l in labels], fontsize=10)
    ax.set_xlabel("LLM Judge Label", fontsize=12)
    ax.set_ylabel("Human Label", fontsize=12)
    ax.set_title(
        f"LLM Judge Validation (n={n})\n"
        f"Agreement={100*accuracy:.1f}%  Cohen's κ={kappa:.2f}",
        fontsize=12,
    )

    fig.colorbar(im, ax=ax, label="Row-normalized frequency", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="Generate annotation sheet for human labeling")
    p_export.add_argument("--predictions", required=True, help="LLM judge predictions CSV")
    p_export.add_argument("--n_samples", type=int, default=100)
    p_export.add_argument("--out_dir", default="data/maira2_judge_validation")

    p_analyze = sub.add_parser("analyze", help="Analyze human vs LLM agreement")
    p_analyze.add_argument("--annotated", required=True, help="Annotated CSV with human_label filled in")

    args = parser.parse_args()
    if args.command == "export":
        cmd_export(args)
    elif args.command == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
