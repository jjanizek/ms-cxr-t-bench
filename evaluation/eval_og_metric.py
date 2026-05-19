"""Apples-to-apples eval against the original MAIRA-2 judge metric.

Generates reports on val+test MS-CXR-T pairs using either a LoRA adapter
or the base MAIRA-2, then scores via the ORIGINAL 3-class judge prompt
(no "none" option, ambiguous defaults to "stable"). Computes per-finding
balanced accuracy averaged across findings — the same macro_acc the
April-16 baseline used (0.365 for specific-prompt MAIRA-2).

Usage:
    # Trained model
    python evaluation/eval_og_metric.py \\
        --config configs/maira2_grpo_lora_dev_v3.yaml \\
        --adapter checkpoints/maira2_grpo_lora_dev_v3/step_000199 \\
        --tag v3_step199

    # Baseline (no LoRA) on the same eval set
    python evaluation/eval_og_metric.py \\
        --config configs/maira2_grpo_lora_dev_v3.yaml \\
        --baseline --tag baseline_valtest
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
    FINDING_DISPLAY, _safe_parse, build_inputs,
    load_model_and_processor, to_device,
)
from data.dataset import FINDINGS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
for noisy in ("httpx", "openai._base_client"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------- OG 3-class judge (copied verbatim from pre-edit llm_judge_temporal.py)
OG_LABEL_MAP = {"improving": 0, "stable": 1, "worsening": 2}
OG_LABEL_NAMES = ["improving", "stable", "worsening"]

OG_SYSTEM_PROMPT = """\
You are an expert radiologist. You will be given a radiology report describing \
chest X-ray findings, and a specific clinical finding to evaluate. Your task is \
to determine whether that finding is IMPROVING, STABLE, or WORSENING based on \
the report's temporal language.

Rules:
- If the report describes the finding as getting better, resolving, decreasing, \
or improved compared to prior, classify as IMPROVING.
- If the report describes the finding as unchanged, similar, stable, or \
persistent without change, classify as STABLE.
- If the report describes the finding as getting worse, increasing, new, \
progressing, or worsened compared to prior, classify as WORSENING.
- If the report does not mention the finding or temporal change at all, use \
your best judgment based on available context. If truly ambiguous, classify \
as STABLE.

Respond with EXACTLY this JSON format (no other text):
{"classification": "improving" | "stable" | "worsening", "reasoning": "one sentence explanation"}
"""

OG_USER_TEMPLATE = """\
Report:
{report}

Finding to evaluate: {finding}

Classify the temporal progression of {finding} as improving, stable, or worsening."""


def og_parse(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
        cls = parsed.get("classification", "").lower().strip()
        if cls in OG_LABEL_MAP:
            return cls
    except json.JSONDecodeError:
        pass
    low = text.lower()
    for label in ["improving", "worsening", "stable"]:
        if label in low:
            return label
    return "stable"


class AsyncOGJudge:
    def __init__(self, model="gpt-4.1-mini", concurrency=16):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
        self.model = model
        self.concurrency = concurrency

    async def grade_one(self, sem, report, finding):
        display = FINDING_DISPLAY.get(finding, finding)
        user = OG_USER_TEMPLATE.format(report=report, finding=display)
        async with sem:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": OG_SYSTEM_PROMPT},
                              {"role": "user", "content": user}],
                    temperature=0, max_tokens=150,
                )
                return og_parse(resp.choices[0].message.content)
            except Exception as e:
                logger.warning("Judge API error: %s", e)
                return "stable"

    async def grade_many(self, items):
        sem = asyncio.Semaphore(self.concurrency)
        return await asyncio.gather(*(self.grade_one(sem, r, f) for r, f in items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit with --baseline)")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--tag", required=True, help="Output filename tag")
    ap.add_argument("--prompts_subset", default="val_test",
                    choices=["val", "test", "val_test"])
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="Cap number of pairs evaluated (random subsample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/maira2_og_metric_eval")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = cfg["model"]["device"]
    images_root = Path(cfg["data"]["images_root"])
    prompts_dir = Path(cfg["data"]["prompts_dir"])
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    splits_to_load = ["val", "test"] if args.prompts_subset == "val_test" else [args.prompts_subset]
    for split in splits_to_load:
        with open(prompts_dir / f"{split}.json") as f:
            pool = json.load(f)
        pairs.extend(pool["pair"])
    logger.info("Loaded %d pair prompts from %s", len(pairs), splits_to_load)
    if args.max_pairs and args.max_pairs < len(pairs):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(pairs), size=args.max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(idx.tolist())]
        logger.info("Subsampled to %d pairs (seed=%d)", len(pairs), args.seed)

    model, processor = load_model_and_processor(cfg)
    if not args.baseline:
        if not args.adapter:
            raise SystemExit("--adapter required without --baseline")
        model.load_adapter(args.adapter, adapter_name="default", is_trainable=False)
        logger.info("Loaded adapter %s", args.adapter)
    model.eval()

    # Force "specific" prompt mode to match the OG eval comparison.
    cfg["prompting"]["prompt_mode"] = "specific"

    reports = []
    adapter_ctx = (model.disable_adapter() if args.baseline else _Null())
    with adapter_ctx, torch.inference_mode():
        for p in tqdm(pairs, desc="generate"):
            inputs = to_device(build_inputs(processor, p, cfg, images_root), device)
            out = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id
                              or processor.tokenizer.eos_token_id,
            )
            T_p = inputs["input_ids"].shape[-1]
            text = processor.decode(out[0][T_p:], skip_special_tokens=True).lstrip()
            reports.append({"prompt": p, "report": _safe_parse(processor, text)})

    judge = AsyncOGJudge()
    judge_items = [(r["report"], r["prompt"]["finding"]) for r in reports]
    preds = asyncio.run(judge.grade_many(judge_items))

    df = pd.DataFrame([{
        "current_dicom": r["prompt"]["current_dicom"],
        "prior_dicom": r["prompt"]["prior_dicom"],
        "finding": r["prompt"]["finding"],
        "gt_progression": r["prompt"]["gt_progression"],
        "predicted": pred,
        "report": r["report"],
    } for r, pred in zip(reports, preds)])

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    df.to_csv(out_dir / f"predictions_{args.tag}_{timestamp}.csv", index=False)

    # Per-finding macro_acc (== balanced_accuracy = mean per-class recall)
    summary = {
        "tag": args.tag,
        "adapter": args.adapter,
        "baseline": args.baseline,
        "n_total": len(df),
        "splits": splits_to_load,
        "per_finding": [],
    }
    accs = []
    for finding in FINDINGS:
        sub = df[df["finding"] == finding]
        if len(sub) == 0:
            continue
        y_true = sub["gt_progression"].map(OG_LABEL_MAP).values
        y_pred = sub["predicted"].map(OG_LABEL_MAP).values
        # OG judge always returns one of 3 classes, but safety: drop unmapped
        mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
        y_true = y_true[mask].astype(int); y_pred = y_pred[mask].astype(int)
        macro_acc = float(balanced_accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
        accs.append(macro_acc)
        summary["per_finding"].append({
            "finding": finding, "n": int(mask.sum()),
            "macro_acc": macro_acc, "confusion_matrix": cm,
        })
        logger.info("  %-20s n=%3d  macro_acc=%.3f", finding, int(mask.sum()), macro_acc)
    summary["average_macro_acc"] = float(np.mean(accs)) if accs else 0.0
    logger.info("AVERAGE macro_acc = %.3f", summary["average_macro_acc"])

    out_path = out_dir / f"summary_{args.tag}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved to %s", out_path)


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
