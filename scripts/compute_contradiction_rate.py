"""Compute the clinical "direct contradiction" rate from saved predictions.

Labels are ordered: improving (0) < stable (1) < worsening (2).

  - exact:       GT == pred                     (|gt - pred| == 0)
  - adjacent:    off by one category            (|gt - pred| == 1)
  - contradiction: improving ↔ worsening        (|gt - pred| == 2)

The "direct contradiction" rate is the clinically most worrying error:
the model flips the direction of change. This is the metric the attending
would care about when saying "worsening or stable is fine, but not improving".

Operates on any CSV with columns: finding, ground_truth (or gt_label), predicted
(or pred_label). Currently only MAIRA-2 outputs have per-sample predictions
saved; discriminative models would need a re-eval pass to produce these.

Usage:
    # Auto-find all available prediction CSVs and compute per-model
    python scripts/compute_contradiction_rate.py

    # Specific files
    python scripts/compute_contradiction_rate.py \
        --predictions results/maira2_judge/predictions_*.csv \
                      results/maira2_judge_specific/predictions_*.csv
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_MAP = {"improving": 0, "stable": 1, "worsening": 2}
LABEL_NAMES = ["improving", "stable", "worsening"]


def load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names across different generators
    if "gt_label" in df.columns:
        df["gt_label"] = df["gt_label"].astype(int)
    elif "ground_truth" in df.columns:
        df["gt_label"] = df["ground_truth"].str.lower().map(LABEL_MAP).astype(int)
    else:
        raise ValueError(f"{csv_path}: no ground_truth / gt_label column")

    if "pred_label" in df.columns:
        df["pred_label"] = df["pred_label"].astype(int)
    elif "predicted" in df.columns:
        df["pred_label"] = df["predicted"].str.lower().map(LABEL_MAP).astype(int)
    else:
        raise ValueError(f"{csv_path}: no predicted / pred_label column")

    if "finding" not in df.columns:
        raise ValueError(f"{csv_path}: no finding column")
    return df


def compute_error_breakdown(gt: np.ndarray, pred: np.ndarray) -> dict:
    diff = np.abs(gt - pred)
    n = len(gt)
    if n == 0:
        return {"n": 0, "exact": 0.0, "adjacent": 0.0, "contradiction": 0.0}
    return {
        "n": int(n),
        "exact": float((diff == 0).mean()),
        "adjacent": float((diff == 1).mean()),
        "contradiction": float((diff == 2).mean()),
    }


def analyze_predictions(df: pd.DataFrame, label: str) -> dict:
    overall = compute_error_breakdown(df["gt_label"].values, df["pred_label"].values)
    per_finding = {}
    for finding, sub in df.groupby("finding"):
        per_finding[finding] = compute_error_breakdown(
            sub["gt_label"].values, sub["pred_label"].values,
        )
    # Directional split: which direction do the contradictions flow?
    is_contra = np.abs(df["gt_label"] - df["pred_label"]) == 2
    contra_rows = df[is_contra]
    gt_improving_pred_worsening = int(
        ((contra_rows["gt_label"] == 0) & (contra_rows["pred_label"] == 2)).sum()
    )
    gt_worsening_pred_improving = int(
        ((contra_rows["gt_label"] == 2) & (contra_rows["pred_label"] == 0)).sum()
    )
    return {
        "label": label,
        "overall": overall,
        "per_finding": per_finding,
        "contradiction_direction": {
            "GT_improving__pred_worsening": gt_improving_pred_worsening,
            "GT_worsening__pred_improving": gt_worsening_pred_improving,
        },
    }


def format_row(label: str, b: dict) -> str:
    if b["n"] == 0:
        return f"  {label:30s}  n=   0"
    return (
        f"  {label:30s}  n={b['n']:4d}  "
        f"exact={100*b['exact']:5.1f}%  "
        f"adj={100*b['adjacent']:5.1f}%  "
        f"contradiction={100*b['contradiction']:5.1f}%"
    )


def print_report(results: list[dict]) -> None:
    for r in results:
        print(f"\n=== {r['label']} ===")
        print(format_row("OVERALL", r["overall"]))
        print(
            f"    direction: {r['contradiction_direction']['GT_improving__pred_worsening']} "
            f"(GT improving → pred worsening)  +  "
            f"{r['contradiction_direction']['GT_worsening__pred_improving']} "
            f"(GT worsening → pred improving)"
        )
        print("    Per finding:")
        for finding, b in sorted(r["per_finding"].items()):
            print("  " + format_row(finding, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", nargs="+", default=None,
        help="One or more predictions CSVs. If omitted, auto-discover under results/.",
    )
    parser.add_argument("--out", default="results/contradiction_rates.json")
    args = parser.parse_args()

    if args.predictions is None:
        candidates = sorted(set(
            glob.glob("results/maira2_judge*/predictions_*.csv")
        ))
        if not candidates:
            print("No predictions CSVs found under results/. "
                  "Pass --predictions explicitly.")
            return
        args.predictions = candidates

    results = []
    for path in args.predictions:
        path = Path(path)
        if not path.exists():
            print(f"skipping missing: {path}")
            continue
        df = load_predictions(path)
        # Nice label: "maira2_judge" or "maira2_judge_specific"
        label = path.parent.name
        r = analyze_predictions(df, label)
        r["path"] = str(path)
        results.append(r)

    print_report(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON written to {out_path}")


if __name__ == "__main__":
    main()
