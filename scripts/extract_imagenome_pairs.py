"""Extract temporal image pairs from Chest ImaGenome silver dataset.

Scans all scene graphs in scene_graph.zip and extracts (prior, current) image pairs
with progression labels (improving/stable/worsening) for the 5 MS-CXR-T findings.

Label aggregation: For each (curr_dicom, prior_dicom, finding), collect all
relationship-level progression votes across anatomical regions. Take majority vote;
skip pairs where no single label has a strict majority.

Outputs:
  data/imagenome_pairs/pairs_train.csv  — training pairs (ImaGenome train split subjects)
  data/imagenome_pairs/pairs_val.csv    — validation pairs (ImaGenome val split subjects)
  data/imagenome_pairs/pairs_stats.json — per-finding counts and label distributions

Also outputs:
  data/imagenome_pairs/download_images.sh — wget script for all unique images needed
  data/imagenome_pairs/dicom_to_path.csv  — mapping dicom_id → physionet path

Usage:
    python scripts/extract_imagenome_pairs.py
    python scripts/extract_imagenome_pairs.py --scene_graph_zip /path/to/scene_graph.zip
"""
import argparse
import json
import logging
import os
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Finding mappings
# ---------------------------------------------------------------------------
# Attribute string → canonical finding name (matching MS-CXR-T FINDINGS)
FINDING_ATTR_MAP = {
    # anatomicalfinding type
    "anatomicalfinding|yes|consolidation":              "consolidation",
    "anatomicalfinding|yes|pleural effusion":           "pleural_effusion",
    "anatomicalfinding|yes|pneumothorax":               "pneumothorax",
    "anatomicalfinding|yes|pulmonary edema/hazy opacity": "edema",
    # disease type
    "disease|yes|pneumonia":                            "pneumonia",
}

FINDINGS = ["consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax"]

# Relationship name → progression label
PROGRESSION_MAP = {
    "comparison|yes|improved":   "improving",
    "comparison|yes|worsened":   "worsening",
    "comparison|yes|no change":  "stable",
}

LABEL_MAP = {"improving": 0, "stable": 1, "worsening": 2}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_pairs_from_zip(zip_path: str) -> list[dict]:
    """Scan all scene graphs and extract temporal pairs.

    Returns a list of dicts:
        curr_dicom_id, prior_dicom_id, patient_id, curr_study_id, finding, progression
    where progression is a majority-vote label (ambiguous pairs are dropped).
    """
    # (curr, prior, finding) → Counter of progressions
    votes: dict[tuple, Counter] = defaultdict(Counter)
    # dicom_id → (patient_id, study_id) — needed to construct download URLs
    dicom_meta: dict[str, tuple] = {}

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith("_SceneGraph.json")]
        logger.info("Processing %d scene graphs...", len(names))

        for name in tqdm(names, desc="Scanning scene graphs"):
            with zf.open(name) as f:
                sg = json.load(f)

            curr_id = sg["image_id"]
            patient_id = sg.get("patient_id")
            study_id = sg.get("study_id")
            viewpoint = sg.get("viewpoint", "")

            # Store metadata for URL generation
            if curr_id not in dicom_meta:
                dicom_meta[curr_id] = (patient_id, study_id)

            # Only use frontal views (AP) to match MS-CXR-T convention
            if viewpoint not in ("AP", "PA"):
                continue

            for rel in sg.get("relationships", []):
                # Parse prior dicom_id from object_id (format: "{dicom_id}_{anatomy}")
                obj_id = rel.get("object_id", "")
                if "_" not in obj_id:
                    continue
                prior_id = obj_id.split("_")[0]
                if prior_id == curr_id:
                    continue

                # Parse finding(s) from relationship attributes
                rel_attrs = rel.get("attributes", [])
                findings_here = set()
                for attr in rel_attrs:
                    finding = FINDING_ATTR_MAP.get(attr)
                    if finding:
                        findings_here.add(finding)
                if not findings_here:
                    continue

                # Parse progression from relationship_names
                rel_names = rel.get("relationship_names", [])
                progressions_here = set()
                for rn in rel_names:
                    prog = PROGRESSION_MAP.get(rn)
                    if prog:
                        progressions_here.add(prog)
                if not progressions_here:
                    continue

                # Vote: add one vote per (finding, progression) combination found
                for finding in findings_here:
                    for prog in progressions_here:
                        votes[(curr_id, prior_id, finding)][prog] += 1

    # Majority vote aggregation
    records = []
    for (curr_id, prior_id, finding), counter in votes.items():
        total = sum(counter.values())
        best_prog, best_count = counter.most_common(1)[0]
        # Require strict majority (>50%) to keep label
        if best_count / total > 0.5:
            patient_id, curr_study_id = dicom_meta.get(curr_id, (None, None))
            records.append({
                "curr_dicom_id": curr_id,
                "prior_dicom_id": prior_id,
                "patient_id": patient_id,
                "curr_study_id": curr_study_id,
                "finding": finding,
                "progression": best_prog,
                "label": LABEL_MAP[best_prog],
                "vote_count": best_count,
                "total_votes": total,
            })

    return records


def build_dicom_to_path(records: list[dict], split_csvs: list[str]) -> dict[str, str]:
    """Build dicom_id → relative MIMIC-CXR path from ImaGenome split files.

    The split CSVs have columns: subject_id, study_id, dicom_id, path (DICOM path)
    We derive the JPG path: files/p{subj[:2]}/p{subj}/s{study}/{dicom}.jpg
    """
    dicom_to_path = {}
    for csv_path in split_csvs:
        if not Path(csv_path).exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            dicom_id = str(row["dicom_id"])
            subject_id = str(int(row["subject_id"]))
            study_id = str(int(row["study_id"]))
            prefix = "p" + subject_id[:2]
            jpg_path = f"files/{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
            dicom_to_path[dicom_id] = jpg_path
    return dicom_to_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_graph_zip",
        default="/data/chest-imagenome/1.0.0/silver_dataset/scene_graph.zip",
    )
    parser.add_argument(
        "--splits_dir",
        default="/data/chest-imagenome/1.0.0/silver_dataset/splits",
    )
    parser.add_argument("--out_dir", default="data/imagenome_pairs")
    parser.add_argument(
        "--physionet_base",
        default="https://physionet.org/files/mimic-cxr-jpg/2.1.0",
    )
    parser.add_argument("--physionet_user", default="jjanizek")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ImaGenome subject-level split assignments
    splits_dir = Path(args.splits_dir)
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "valid.csv")
    avoid_df = pd.read_csv(splits_dir / "images_to_avoid.csv")

    train_subjects = set(train_df["subject_id"].astype(str))
    val_subjects = set(val_df["subject_id"].astype(str))
    avoid_dicoms = set(avoid_df["dicom_id"].astype(str))
    logger.info(
        "Train subjects: %d, Val subjects: %d, Avoid images: %d",
        len(train_subjects), len(val_subjects), len(avoid_dicoms),
    )

    # Build dicom_id → JPG path mapping from split CSVs
    dicom_to_path = build_dicom_to_path(
        [],  # records not needed here
        [
            str(splits_dir / "train.csv"),
            str(splits_dir / "valid.csv"),
            str(splits_dir / "test.csv"),
        ],
    )
    logger.info("dicom_to_path entries: %d", len(dicom_to_path))

    # Extract all temporal pairs
    records = extract_pairs_from_zip(args.scene_graph_zip)
    logger.info("Total temporal pairs (after majority vote): %d", len(records))

    df = pd.DataFrame(records)

    # Remove images_to_avoid
    before = len(df)
    df = df[~df["curr_dicom_id"].isin(avoid_dicoms) & ~df["prior_dicom_id"].isin(avoid_dicoms)]
    logger.info("Removed %d pairs with avoid-listed images; %d remain", before - len(df), len(df))

    # Assign split by patient_id
    df["patient_id_str"] = df["patient_id"].astype(str)
    df_train = df[df["patient_id_str"].isin(train_subjects)].copy()
    df_val = df[df["patient_id_str"].isin(val_subjects)].copy()

    # Stats
    stats = {}
    for finding in FINDINGS:
        entry = {}
        for split_name, split_df in [("train", df_train), ("val", df_val)]:
            sub = split_df[split_df["finding"] == finding]
            entry[split_name] = {
                "total": len(sub),
                "improving": int((sub["progression"] == "improving").sum()),
                "stable":    int((sub["progression"] == "stable").sum()),
                "worsening": int((sub["progression"] == "worsening").sum()),
            }
        stats[finding] = entry
        logger.info(
            "  %-20s  train=%5d (I=%d S=%d W=%d)  val=%4d (I=%d S=%d W=%d)",
            finding,
            entry["train"]["total"], entry["train"]["improving"],
            entry["train"]["stable"], entry["train"]["worsening"],
            entry["val"]["total"], entry["val"]["improving"],
            entry["val"]["stable"], entry["val"]["worsening"],
        )

    # Save CSVs
    df_train.to_csv(out_dir / "pairs_train.csv", index=False)
    df_val.to_csv(out_dir / "pairs_val.csv", index=False)
    with open(out_dir / "pairs_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved pairs to %s", out_dir)

    # Generate targeted download script
    all_needed_dicoms = (
        set(df_train["curr_dicom_id"])
        | set(df_train["prior_dicom_id"])
        | set(df_val["curr_dicom_id"])
        | set(df_val["prior_dicom_id"])
    )
    logger.info("Unique images needed: %d", len(all_needed_dicoms))

    # Write URL list: one line per image — "url dest_path"
    # Used by the parallel download script via xargs
    url_lines = []
    missing_path = 0
    for dicom_id in sorted(all_needed_dicoms):
        rel_path = dicom_to_path.get(dicom_id)
        if rel_path is None:
            missing_path += 1
            continue
        url = f"{args.physionet_base}/{rel_path}"
        url_lines.append(f"{url}\t{dicom_id}.jpg")

    if missing_path:
        logger.warning("No path found for %d dicom_ids (not in split CSVs)", missing_path)

    url_list_path = out_dir / "image_urls.tsv"
    with open(url_list_path, "w") as f:
        f.write("\n".join(url_lines) + "\n")
    logger.info("URL list saved to %s (%d entries)", url_list_path, len(url_lines))

    # Parallel download script: uses xargs -P for concurrency
    dl_script_content = f"""#!/bin/bash
# Download {len(url_lines)} MIMIC-CXR-JPG images for Chest ImaGenome training.
# Uses 16 parallel wget processes — typically 30-60 min on a fast connection.
# Safe to interrupt and re-run (wget -nc skips already-downloaded files).
#
# Usage: bash data/imagenome_pairs/download_images.sh

read -sp "PhysioNet password for {args.physionet_user}: " PW
echo

DEST=/data/imagenome_images
mkdir -p "$DEST"

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
URL_LIST="$SCRIPT_DIR/image_urls.tsv"

echo "Downloading $(wc -l < "$URL_LIST") images to $DEST ..."

# Each line: url<TAB>filename
# xargs -P 16 runs 16 parallel downloads
cat "$URL_LIST" | xargs -P 16 -I{{}} bash -c '
  URL=$(echo {{}} | cut -f1)
  FNAME=$(echo {{}} | cut -f2)
  DEST_FILE="'"$DEST"'/$FNAME"
  if [ -f "$DEST_FILE" ]; then
    exit 0
  fi
  wget -q --user="{args.physionet_user}" --password="'"$PW"'" -O "$DEST_FILE" "$URL" || {{
    rm -f "$DEST_FILE"
    echo "FAIL: $FNAME"
  }}
'

echo "Done. $(ls "$DEST" | wc -l) files in $DEST"
"""

    dl_script = out_dir / "download_images.sh"
    with open(dl_script, "w") as f:
        f.write(dl_script_content)
    os.chmod(dl_script, 0o755)
    logger.info("Download script saved to %s", dl_script)

    # Save dicom_to_path lookup for all needed
    path_records = [
        {"dicom_id": d, "rel_path": dicom_to_path.get(d, "")}
        for d in sorted(all_needed_dicoms)
    ]
    pd.DataFrame(path_records).to_csv(out_dir / "dicom_to_path.csv", index=False)


if __name__ == "__main__":
    main()
