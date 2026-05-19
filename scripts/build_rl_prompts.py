"""Build pair + single prompt pools for MAIRA-2 GRPO training.

Two prompt types are materialized at the subject level using existing splits
in data/splits/split_seed{seed}.json:

  pair   — (current_dicom, prior_dicom, finding, gt_progression) from MS-CXR-T.
           Reward: judge classification matches gt_progression.
  single — (current_dicom, finding) with prior_frontal=None at training time.
           Reward: judge says makes_comparison == False (no hallucinated prior).

Single-image prompts are constructed from the same current_dicoms used in the
pair pool, crossed with the 5 findings.  This keeps the image distribution
matched between the two prompt types (so the LM can't tell pair vs single from
image statistics).

Usage:
    python scripts/build_rl_prompts.py --seed 42
    python scripts/build_rl_prompts.py --seed 42 --out_dir data/rl_prompts/seed42
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import load_labels, FINDINGS, dicom_id_to_filename

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_pair_prompts(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, row in df.iterrows():
        out.append({
            "prompt_id": f"pair__{Path(row['dicom_id']).name}__"
                          f"{Path(row['previous_dicom_id']).name}__{row['finding']}",
            "prompt_type": "pair",
            "finding": row["finding"],
            "current_dicom": row["dicom_id"],
            "prior_dicom": row["previous_dicom_id"],
            "current_file": dicom_id_to_filename(row["dicom_id"]),
            "prior_file": dicom_id_to_filename(row["previous_dicom_id"]),
            "gt_progression": row["progression"],
            "subject_id": int(row["subject_id"]),
        })
    return out


def build_single_prompts(df: pd.DataFrame) -> list[dict]:
    # Use every current_dicom seen in pair rows, crossed with all 5 findings.
    # Note: we DON'T cross with the prior_dicom too — we want the single pool to
    # share the same image distribution as the pair pool's "current" images.
    seen = {}
    for _, row in df.iterrows():
        seen[row["dicom_id"]] = int(row["subject_id"])

    out = []
    for dicom_id, subject_id in seen.items():
        for finding in FINDINGS:
            out.append({
                "prompt_id": f"single__{Path(dicom_id).name}__{finding}",
                "prompt_type": "single",
                "finding": finding,
                "current_dicom": dicom_id,
                "prior_dicom": None,
                "current_file": dicom_id_to_filename(dicom_id),
                "prior_file": None,
                "gt_progression": None,
                "subject_id": subject_id,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/raw/ms_cxr_t_labels.csv")
    ap.add_argument("--images_root", default="data/raw/images")
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=None,
                    help="Default: data/rl_prompts/seed{seed}")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"data/rl_prompts/seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    split_path = Path(args.splits_dir) / f"split_seed{args.seed}.json"
    with open(split_path) as f:
        split = json.load(f)
    logger.info("Loaded split %s: train=%d val=%d test=%d subjects",
                split_path, len(split["train"]), len(split["val"]), len(split["test"]))

    df = load_labels(args.labels, images_root=args.images_root)
    logger.info("Loaded %d (pair, finding) rows after image-existence filter", len(df))

    for split_name in ("train", "val", "test"):
        subj_set = set(split[split_name])
        df_s = df[df["subject_id"].isin(subj_set)].copy()
        pair_prompts = build_pair_prompts(df_s)
        single_prompts = build_single_prompts(df_s)

        out_path = out_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump({
                "split": split_name, "seed": args.seed,
                "pair": pair_prompts, "single": single_prompts,
            }, f, indent=2)

        logger.info("  %-5s  pair=%4d  single=%4d  → %s",
                    split_name, len(pair_prompts), len(single_prompts), out_path)


if __name__ == "__main__":
    main()
