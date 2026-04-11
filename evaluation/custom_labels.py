"""Evaluation on our supplementary custom label set.

Custom labels live in data/custom_labels/ and are loaded by this module.
Evaluation mirrors Protocol A (linear probe) on the custom annotations.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from evaluation.temporal_cls import run_protocol_a

logger = logging.getLogger(__name__)

CUSTOM_LABEL_DIR = Path("data/custom_labels")


def load_custom_labels(label_file: Optional[Path] = None) -> pd.DataFrame:
    """Load the supplementary custom label CSV.

    Expected columns: subject_id, study_id_1, study_id_2, finding, label
    """
    if label_file is None:
        candidates = sorted(CUSTOM_LABEL_DIR.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No CSV files found in {CUSTOM_LABEL_DIR}. "
                "Generate custom labels first."
            )
        label_file = candidates[-1]
        logger.info("Using custom label file: %s", label_file)
    return pd.read_csv(label_file)


def run_custom_label_eval(
    features: dict,
    seed: int,
    label_file: Optional[Path] = None,
    **probe_kwargs,
) -> list[dict]:
    """Evaluate on each finding in the custom label set."""
    df = load_custom_labels(label_file)
    findings = df["finding"].unique().tolist()
    results = []
    for finding in findings:
        subset = df[df["finding"] == finding]
        # features dict expected keyed by finding
        if finding not in features:
            logger.warning("No features for finding %s, skipping", finding)
            continue
        result = run_protocol_a(features[finding], finding, seed, **probe_kwargs)
        result["label_set"] = "custom"
        results.append(result)
    return results
