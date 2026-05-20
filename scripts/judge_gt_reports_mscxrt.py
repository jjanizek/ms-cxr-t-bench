"""Judge the *ground-truth* MIMIC-CXR reports for the MS-CXR-T val+test pairs.

Answers two questions at once:
  1. What's the upper bound of our OG 3-class judge on MS-CXR-T? If we hand
     the judge the real radiologist's report for the current study (which
     described the temporal change explicitly) and grade vs the MS-CXR-T
     temporal label, how often does the judge agree?
  2. Where does judge↔label disagreement concentrate? Per-class confusion
     and per-finding agreement.

Reports come from Chest ImaGenome's processed-sentences dump, grouped by
(subject_id, rad_id) — rad_id is the MIMIC study_id (sXXXXXXXX). MS-CXR-T
pair prompts in data/rl_prompts/seed42 carry current_dicom like
p10/p10056223/s59315493/<dicom> so we parse out the s-id and join.

Usage:
    python scripts/judge_gt_reports_mscxrt.py --tag gt_baseline
"""
import argparse
import asyncio
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.eval_og_metric import AsyncOGJudge, OG_LABEL_MAP
from data.dataset import FINDINGS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
for noisy in ("httpx", "openai._base_client"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

SENTENCES_TSV = "/data/chest-imagenome/1.0.0/silver_dataset/cxr-mimic-v2.0.0-processed-sentences_all.txt"

STUDY_RE = re.compile(r"/s(\d+)/")


def load_reports():
    """Group ImaGenome sentences into one report per (subject_id, rad_id).

    rad_id matches MIMIC study_id; cast to int to match the s-prefix path
    field after stripping the 's'.
    """
    logger.info("Loading sentences from %s ...", SENTENCES_TSV)
    df = pd.read_csv(SENTENCES_TSV, sep="\t", dtype={"subject_id": int, "rad_id": int})
    logger.info("  %d sentence rows", len(df))
    # Some rows are header noise ("FINAL REPORT" etc) but those are part of
    # the report — keep everything as-is. Sort by sent_loc to preserve order.
    df = df.sort_values(["subject_id", "rad_id", "sent_loc"])
    reports = {}
    for (subj, rad), g in df.groupby(["subject_id", "rad_id"], sort=False):
        # sentences come pre-quoted with embedded newlines; just join with space
        text = " ".join(str(s).strip() for s in g["sentence"].tolist() if pd.notna(s))
        # collapse runs of whitespace and stray quotes
        text = re.sub(r"\s+", " ", text).replace('"', '').strip()
        reports[(subj, rad)] = text
    logger.info("  %d unique reports", len(reports))
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gt_baseline")
    ap.add_argument("--prompts_dir", default="data/rl_prompts/seed42")
    ap.add_argument("--out_dir", default="results/maira2_og_metric_eval")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # MS-CXR-T pair prompts (val + test)
    pairs = []
    for split in ["val", "test"]:
        with open(Path(args.prompts_dir) / f"{split}.json") as f:
            pairs.extend(json.load(f)["pair"])
    logger.info("Loaded %d MS-CXR-T pair prompts", len(pairs))

    reports = load_reports()

    # Attach GT report text to each prompt by parsing study_id from current_dicom
    matched = 0
    unmatched_examples = []
    enriched = []
    for p in pairs:
        m = STUDY_RE.search(p["current_dicom"])
        if not m:
            continue
        rad_id = int(m.group(1))
        subj = int(p["subject_id"])
        report = reports.get((subj, rad_id))
        if report is None:
            if len(unmatched_examples) < 3:
                unmatched_examples.append((subj, rad_id, p["current_dicom"]))
            continue
        matched += 1
        enriched.append({**p, "gt_report": report})
    logger.info("Matched GT report for %d / %d pairs", matched, len(pairs))
    if unmatched_examples:
        logger.warning("Unmatched examples: %s", unmatched_examples)

    if not enriched:
        sys.exit("No matched reports; aborting.")

    # Judge each (gt_report, finding) with the OG 3-class judge
    judge = AsyncOGJudge()
    items = [(e["gt_report"], e["finding"]) for e in enriched]
    logger.info("Calling judge on %d items ...", len(items))
    preds = asyncio.run(judge.grade_many(items))

    df = pd.DataFrame([{
        "subject_id": e["subject_id"],
        "current_dicom": e["current_dicom"],
        "prior_dicom": e["prior_dicom"],
        "finding": e["finding"],
        "gt_progression": e["gt_progression"],
        "judge_label": pred,
        "gt_report": e["gt_report"][:1500],  # truncate for CSV readability
    } for e, pred in zip(enriched, preds)])

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    df.to_csv(out_dir / f"gt_report_judge_{args.tag}_{timestamp}.csv", index=False)

    # Per-finding macro_acc + overall + confusion
    summary = {"tag": args.tag, "n_total": len(df), "per_finding": []}
    accs = []
    for finding in FINDINGS:
        sub = df[df["finding"] == finding]
        if len(sub) == 0:
            continue
        y_true = sub["gt_progression"].map(OG_LABEL_MAP).values
        y_pred = sub["judge_label"].map(OG_LABEL_MAP).values
        mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
        y_true = y_true[mask].astype(int)
        y_pred = y_pred[mask].astype(int)
        macro = float(balanced_accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
        accs.append(macro)
        summary["per_finding"].append({
            "finding": finding, "n": int(mask.sum()),
            "macro_acc": macro, "confusion_matrix": cm,
        })
        logger.info("  %-20s n=%3d  judge↔gt macro_acc=%.3f", finding, int(mask.sum()), macro)
    summary["average_macro_acc"] = float(np.mean(accs)) if accs else 0.0
    logger.info("AVERAGE judge↔gt macro_acc = %.3f", summary["average_macro_acc"])

    # Overall 3-class confusion
    y_true_all = df["gt_progression"].map(OG_LABEL_MAP).values
    y_pred_all = df["judge_label"].map(OG_LABEL_MAP).values
    mask = ~pd.isna(y_true_all) & ~pd.isna(y_pred_all)
    cm_all = confusion_matrix(
        y_true_all[mask].astype(int), y_pred_all[mask].astype(int), labels=[0, 1, 2]
    )
    summary["overall_confusion_matrix"] = cm_all.tolist()
    summary["overall_confusion_labels"] = ["improving", "stable", "worsening"]
    logger.info("Overall confusion (rows=gt, cols=judge):\n%s\n  (labels: imp / stab / wors)",
                cm_all.tolist())

    out_path = out_dir / f"gt_report_judge_summary_{args.tag}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", out_path)


if __name__ == "__main__":
    main()
