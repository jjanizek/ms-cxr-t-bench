"""Build pair prompts for RL on Chest ImaGenome silver labels.

Reads scripts/extract_imagenome_pairs.py outputs:
  - data/imagenome_pairs/pairs_{train,val}.csv
  - data/imagenome_pairs/dicom_to_path.csv

Filters to pairs where BOTH the current and prior JPEGs exist on disk under
the configured MIMIC-CXR-JPG root, then writes pair-prompts in the same
format as scripts/build_rl_prompts.py so they're drop-in for
scripts/rl_maira2_temporal.py.

Usage:
    python scripts/build_imagenome_rl_prompts.py
    python scripts/build_imagenome_rl_prompts.py --images_root /data/mimic-cxr-jpg
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_prompts(pairs_df: pd.DataFrame, path_map: dict[str, str],
                   images_root: Path, available_cache: set[str]) -> list[dict]:
    out = []
    n_missing_path = 0
    n_missing_file = 0
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df),
                       desc="checking pairs", leave=False):
        cur_rel = path_map.get(row["curr_dicom_id"])
        pri_rel = path_map.get(row["prior_dicom_id"])
        if cur_rel is None or pri_rel is None:
            n_missing_path += 1
            continue
        if cur_rel not in available_cache or pri_rel not in available_cache:
            n_missing_file += 1
            continue
        out.append({
            "prompt_id": f"ig__{row['curr_dicom_id']}__"
                         f"{row['prior_dicom_id']}__{row['finding']}",
            "prompt_type": "pair",
            "finding": row["finding"],
            "current_dicom": row["curr_dicom_id"],
            "prior_dicom": row["prior_dicom_id"],
            "current_file": cur_rel,
            "prior_file": pri_rel,
            "gt_progression": row["progression"],
            "subject_id": int(row["patient_id"]),
        })
    logger.info("  kept=%d  missing_path=%d  missing_file=%d",
                len(out), n_missing_path, n_missing_file)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", default="data/imagenome_pairs")
    ap.add_argument("--images_root", default="/data/mimic-cxr-jpg")
    ap.add_argument("--out_dir", default="data/rl_prompts/imagenome")
    args = ap.parse_args()

    pairs_dir = Path(args.pairs_dir)
    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading dicom→path map ...")
    path_df = pd.read_csv(pairs_dir / "dicom_to_path.csv")
    path_map = dict(zip(path_df["dicom_id"], path_df["rel_path"]))
    logger.info("  %d dicom_ids in path map", len(path_map))

    # Walk MIMIC-CXR-JPG once and cache existing rel paths — far faster than
    # 100k+ individual stat() calls.
    logger.info("Scanning %s for existing JPEGs ...", images_root)
    available = set()
    files_root = images_root / "files"
    for jpg in tqdm(files_root.rglob("*.jpg"), desc="scan", unit="file"):
        rel = str(jpg.relative_to(images_root))
        available.add(rel)
    logger.info("  %d JPEGs present on disk", len(available))

    for split in ("train", "val"):
        csv_path = pairs_dir / f"pairs_{split}.csv"
        df = pd.read_csv(csv_path)
        logger.info("%s: %d pair-finding rows", split, len(df))
        prompts = build_prompts(df, path_map, images_root, available)

        # Per-finding stats so we know the class/finding balance the trainer sees
        per_finding = {}
        for f, g in pd.DataFrame(prompts).groupby("finding"):
            per_finding[f] = {
                "n": len(g),
                "by_progression": g["gt_progression"].value_counts().to_dict(),
            }

        out_path = out_dir / f"{split}.json"
        with open(out_path, "w") as fout:
            json.dump({
                "split": split, "source": "chest_imagenome_silver",
                "pair": prompts,
                "single": [],   # not used for ImaGenome
                "per_finding": per_finding,
            }, fout, indent=2)
        logger.info("  → %s (%d pairs)", out_path, len(prompts))
        for f, s in per_finding.items():
            logger.info("    %-20s n=%5d  %s", f, s["n"], s["by_progression"])


if __name__ == "__main__":
    main()
