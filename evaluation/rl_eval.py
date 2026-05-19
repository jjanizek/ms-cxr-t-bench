"""Standalone evaluation of a MAIRA-2 LoRA checkpoint on the held-out test set.

Loads a trained adapter from --adapter, runs greedy decode on test pair + single
prompts, and scores with the LLM judge. Saves predictions + summary metrics.

Usage:
    python evaluation/rl_eval.py \\
        --config configs/maira2_grpo_lora.yaml \\
        --adapter checkpoints/maira2_grpo_lora/step_001999 \\
        --split test
"""
import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rl_maira2_temporal import (
    AsyncJudge, FINDING_DISPLAY, LABEL_MAP,
    build_inputs, load_model_and_processor, to_device,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit with --baseline)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--max_pair", type=int, default=None)
    ap.add_argument("--max_single", type=int, default=None)
    ap.add_argument("--baseline", action="store_true",
                    help="Eval base model with adapter disabled (no LoRA)")
    ap.add_argument("--out_dir", default="results/maira2_grpo_lora_eval")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["model"]["device"]
    images_root = Path(cfg["data"]["images_root"])
    prompts_dir = Path(cfg["data"]["prompts_dir"])
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with open(prompts_dir / f"{args.split}.json") as f:
        pool = json.load(f)
    pair_p = pool["pair"][: args.max_pair] if args.max_pair else pool["pair"]
    single_p = pool["single"][: args.max_single] if args.max_single else pool["single"]
    logger.info("%s split: pair=%d single=%d", args.split, len(pair_p), len(single_p))

    model, processor = load_model_and_processor(cfg)
    if not args.baseline:
        if not args.adapter:
            raise SystemExit("--adapter is required unless --baseline is set")
        adapter_state = Path(args.adapter)
        model.load_adapter(str(adapter_state), adapter_name="default", is_trainable=False)
        logger.info("Loaded adapter from %s", adapter_state)
    model.eval()

    judge = AsyncJudge(model=cfg["judge"]["model"],
                       concurrency=cfg["judge"]["concurrency"])

    all_prompts = pair_p + single_p
    rows = []
    adapter_ctx = (model.disable_adapter() if args.baseline
                   else _Null())
    with adapter_ctx, torch.inference_mode():
        for p in tqdm(all_prompts, desc=args.split):
            inputs = to_device(build_inputs(processor, p, cfg, images_root), device)
            out = model.generate(
                **inputs, do_sample=False,
                max_new_tokens=cfg["sampling"]["max_new_tokens"],
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id
                              or processor.tokenizer.eos_token_id,
            )
            T_p = inputs["input_ids"].shape[-1]
            text = processor.decode(out[0][T_p:], skip_special_tokens=True).lstrip()
            text = processor.convert_output_to_plaintext_or_grounded_sequence(text)
            rows.append({"prompt": p, "report": text if isinstance(text, str) else str(text)})

    judge_items = [(r["report"], r["prompt"]["finding"]) for r in rows]
    judge_outs = asyncio.run(judge.grade_many(judge_items))
    for r, jo in zip(rows, judge_outs):
        r["judge"] = jo

    # Build dataframe
    df = pd.DataFrame([{
        "prompt_id": r["prompt"]["prompt_id"],
        "prompt_type": r["prompt"]["prompt_type"],
        "finding": r["prompt"]["finding"],
        "current_dicom": r["prompt"]["current_dicom"],
        "prior_dicom": r["prompt"]["prior_dicom"],
        "gt_progression": r["prompt"]["gt_progression"],
        "predicted": r["judge"]["classification"],
        "makes_comparison": r["judge"]["makes_comparison"],
        "reasoning": r["judge"]["reasoning"],
        "report": r["report"],
    } for r in rows])

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tag = "baseline" if args.baseline else Path(args.adapter).name
    df.to_csv(out_dir / f"predictions_{tag}_{args.split}_{timestamp}.csv", index=False)

    # Metrics
    summary = {"adapter": tag, "split": args.split, "timestamp": timestamp}

    pair_df = df[df["prompt_type"] == "pair"]
    if len(pair_df):
        y_true = pair_df["gt_progression"].map(LABEL_MAP).values
        y_pred = pair_df["predicted"].map(lambda v: LABEL_MAP.get(v, 3)).values
        summary["pair_n"] = len(pair_df)
        summary["pair_macro_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        summary["pair_exact_acc"] = float((y_true == y_pred).mean())
        summary["pair_per_finding"] = {}
        for f, g in pair_df.groupby("finding"):
            yt = g["gt_progression"].map(LABEL_MAP).values
            yp = g["predicted"].map(lambda v: LABEL_MAP.get(v, 3)).values
            summary["pair_per_finding"][f] = {
                "n": len(g),
                "macro_acc": float(balanced_accuracy_score(yt, yp)),
                "cm": confusion_matrix(yt, yp, labels=[0, 1, 2, 3]).tolist(),
            }

    single_df = df[df["prompt_type"] == "single"]
    if len(single_df):
        summary["single_n"] = len(single_df)
        summary["single_no_comparison_rate"] = float(
            (single_df["makes_comparison"] == False).mean()
        )
        summary["single_per_finding"] = {
            f: {"n": len(g),
                "no_comparison_rate": float((g["makes_comparison"] == False).mean())}
            for f, g in single_df.groupby("finding")
        }

    out_path = out_dir / f"summary_{tag}_{args.split}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    logger.info("Saved to %s", out_path)


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
