"""Generate and save subject-level train/val/test splits for MS-CXR-T.

Splits at the subject level across all findings combined, then verify per-finding
class balance.  Outputs one JSON per seed to data/splits/.

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --labels data/raw/ms_cxr_t_labels.csv --seeds 42 123 456 789
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import from data module (run from repo root)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import load_labels, FINDINGS


def make_split(df: pd.DataFrame, seed: int, train_frac=0.70, val_frac=0.10) -> dict:
    """Stratified subject-level split.

    Stratification uses the most common (finding, progression) label per subject
    as a proxy.  Returns dict with 'train', 'val', 'test' subject_id lists.
    """
    subjects = df["subject_id"].unique()

    # Majority (finding, progression) label per subject for stratification
    subj_label = (
        df.groupby("subject_id")["progression"]
        .agg(lambda x: x.value_counts().index[0])
        .reindex(subjects)
    )

    test_frac = 1.0 - train_frac - val_frac

    subj_trainval, subj_test = train_test_split(
        subjects,
        test_size=test_frac,
        stratify=subj_label.loc[subjects],
        random_state=seed,
    )
    val_relative = val_frac / (train_frac + val_frac)
    subj_train, subj_val = train_test_split(
        subj_trainval,
        test_size=val_relative,
        stratify=subj_label.loc[subj_trainval],
        random_state=seed,
    )

    return {
        "train": sorted(int(s) for s in subj_train),
        "val": sorted(int(s) for s in subj_val),
        "test": sorted(int(s) for s in subj_test),
    }


def print_split_stats(df: pd.DataFrame, split: dict):
    """Print per-finding class counts for each split partition."""
    for part, subj_list in split.items():
        sub = df[df["subject_id"].isin(subj_list)]
        print(f"\n  {part} ({len(subj_list)} subjects, {len(sub)} pairs):")
        for finding in FINDINGS:
            counts = sub[sub["finding"] == finding]["progression"].value_counts().to_dict()
            total = sum(counts.values())
            print(f"    {finding:20s}: {total:4d} pairs — {counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/raw/ms_cxr_t_labels.csv")
    parser.add_argument("--images_root", default="data/raw/images")
    parser.add_argument("--out_dir", default="data/splits")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789])
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.10)
    args = parser.parse_args()

    df = load_labels(args.labels, images_root=args.images_root)
    print(f"Loaded {len(df)} annotated (pair, finding) rows")
    print(f"  Subjects: {df['subject_id'].nunique()}")
    print(f"  Findings: {df['finding'].value_counts().to_dict()}")
    print(f"  Progression: {df['progression'].value_counts().to_dict()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        split = make_split(df, seed, args.train_frac, args.val_frac)
        counts = {k: len(v) for k, v in split.items()}
        print(f"\nSeed {seed}: subjects {counts}")
        print_split_stats(df, split)

        out_path = out_dir / f"split_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump({"seed": seed, **split}, f, indent=2)
        print(f"  → saved {out_path}")


if __name__ == "__main__":
    main()
