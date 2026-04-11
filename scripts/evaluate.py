"""Eval-only entrypoint (Protocol B/C, sentence similarity, custom labels).

Usage:
    python scripts/evaluate.py --config configs/biovil_t_protocol_b.yaml
    python scripts/evaluate.py --config configs/biovil_t_sentence_sim.yaml
"""
import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    protocol = cfg.get("protocol", "B")
    logger.info("Config: %s | Protocol: %s", args.config, protocol)

    if protocol == "B":
        from evaluation.temporal_cls import run_protocol_b
        # TODO: load features, run SVM
        raise NotImplementedError("Protocol B eval not yet wired up.")
    elif protocol == "sentence_sim":
        from evaluation.sentence_sim import run_sentence_similarity
        # TODO: load text features, run eval
        raise NotImplementedError("Sentence similarity eval not yet wired up.")
    elif protocol == "custom":
        from evaluation.custom_labels import run_custom_label_eval
        raise NotImplementedError("Custom label eval not yet wired up.")
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


if __name__ == "__main__":
    main()
