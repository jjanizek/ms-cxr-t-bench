"""Supervised fine-tuning of MAIRA-2 on (image_pair → full MIMIC-CXR report).

Built to warm-start v7e — i.e. MAIRA-2 with the BioViL-T ensemble vision
tower — out of the cold-start trap where the frozen LM defaults to
predicting "stable" because it can't read the new vision features. SFT
teaches the LM what to actually generate given those features; RL then
refines.

Inputs per training example:
  - Pair prompt: current_dicom + prior_dicom + technique + comparison
    (same format as `rl_maira2_temporal.build_inputs`)
  - Target text: the full radiologist report for the current study,
    reconstructed by joining sentences in ImaGenome's processed-sentences
    dump (the same source the judge-ceiling experiment validated at 0.98
    agreement, so we know reports are faithful).

Loss: token-level cross-entropy on the report tokens only (prompt tokens
masked with -100). Trainable params: whatever PEFT marks via the v7e config
(LM LoRA + projector LoRA + the BioViL-T adapter/prefix/LayerNorm in
modules_to_save).

Usage:
    OPENAI_API_KEY=... python scripts/sft_maira2_temporal.py \
        --config configs/maira2_sft_imagenome_v7e.yaml
"""
import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rl_maira2_temporal import (
    build_inputs, load_model_and_processor, to_device,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
for noisy in ("httpx", "openai._base_client"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


SENTENCES_TSV = "/data/chest-imagenome/1.0.0/silver_dataset/cxr-mimic-v2.0.0-processed-sentences_all.txt"
STUDY_RE = re.compile(r"/s(\d+)/")


def load_reports() -> dict:
    """Group ImaGenome sentences into one full-text report per (subject_id, rad_id).

    Returns a dict keyed by (int subject_id, int rad_id) → joined report string.
    rad_id is MIMIC's study_id without the 's' prefix.
    """
    logger.info("Loading sentences from %s ...", SENTENCES_TSV)
    df = pd.read_csv(SENTENCES_TSV, sep="\t",
                     dtype={"subject_id": int, "rad_id": int})
    logger.info("  %d sentence rows", len(df))
    df = df.sort_values(["subject_id", "rad_id", "sent_loc"])
    reports = {}
    for (subj, rad), g in df.groupby(["subject_id", "rad_id"], sort=False):
        text = " ".join(str(s).strip() for s in g["sentence"].tolist() if pd.notna(s))
        text = re.sub(r"\s+", " ", text).replace('"', "").strip()
        reports[(subj, rad)] = text
    logger.info("  %d unique reports", len(reports))
    return reports


def build_sft_pool(pair_pool: list, reports: dict) -> list:
    """For each pair prompt, attach the full report for the current study.

    Drops pairs whose current report isn't found in the dump.
    """
    out = []
    skipped = 0
    for p in pair_pool:
        # ImaGenome prompts: study_id is only in `current_file` (path).
        # MS-CXR-T prompts: `current_dicom` IS the path. Try both.
        m = STUDY_RE.search(p.get("current_file") or "") \
            or STUDY_RE.search(p.get("current_dicom") or "")
        if not m:
            skipped += 1; continue
        key = (int(p["subject_id"]), int(m.group(1)))
        report = reports.get(key)
        if not report:
            skipped += 1; continue
        out.append({**p, "target_report": report})
    logger.info("Built SFT pool: %d examples (%d skipped, no report match)",
                len(out), skipped)
    return out


def sft_step(model, processor, batch_prompts, cfg, images_root, device):
    """One SFT step: compute CE on target report tokens for each prompt in batch.

    Concatenates prompt+target along the sequence dim, masks loss on prompt
    positions, and averages per-token CE across the batch.
    """
    losses = []
    for prompt in batch_prompts:
        # Build the prompt inputs (vision + text up to the response).
        inp = build_inputs(processor, prompt, cfg, images_root)
        inp = to_device(inp, device)

        # Tokenize the target report (no special tokens; will append to prompt).
        target_text = prompt["target_report"]
        # Cap to keep memory reasonable
        max_tgt = int(cfg["train"].get("max_target_tokens", 256))
        tgt_ids = processor.tokenizer(
            target_text, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=max_tgt,
        )["input_ids"].to(device)

        # Append target to prompt input_ids; build attention mask + labels.
        input_ids = torch.cat([inp["input_ids"], tgt_ids], dim=1)
        attn = torch.cat([
            inp["attention_mask"],
            torch.ones_like(tgt_ids),
        ], dim=1)
        labels = input_ids.clone()
        # Mask loss for the prompt prefix
        labels[:, :inp["input_ids"].shape[1]] = -100

        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            pixel_values=inp.get("pixel_values"),
            image_sizes=inp.get("image_sizes"),
            labels=labels,
            use_cache=False,
        )
        losses.append(out.loss)
    return torch.stack(losses).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None,
                    help="Optional PEFT adapter dir to resume from")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["model"]["device"]
    images_root = Path(cfg["data"]["images_root"])
    prompts_dir = Path(cfg["data"]["prompts_dir"])

    out_dir = Path(cfg["logging"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["checkpoint"]["dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    with open(prompts_dir / "train.json") as f:
        pool = json.load(f)["pair"]
    # Apply same class filter as the matching RL run if requested
    train_classes = cfg["data"].get("train_classes")
    if train_classes:
        before = len(pool)
        pool = [p for p in pool if p.get("gt_progression") in train_classes]
        logger.info("Filtered pair pool by gt_progression %s: %d → %d",
                    train_classes, before, len(pool))

    reports = load_reports()
    sft_pool = build_sft_pool(pool, reports)
    # Optionally subsample
    max_examples = cfg["data"].get("max_examples")
    if max_examples and max_examples < len(sft_pool):
        rng = np.random.default_rng(cfg["data"].get("seed", 42))
        idx = rng.choice(len(sft_pool), size=max_examples, replace=False)
        sft_pool = [sft_pool[int(i)] for i in idx]
        logger.info("Subsampled to %d examples", len(sft_pool))

    # ---- Model ----
    model, processor = load_model_and_processor(cfg)
    if args.resume:
        model.load_adapter(args.resume, adapter_name="default")
        logger.info("Resumed adapter from %s", args.resume)

    # Gradient checkpointing — same setup as the RL script. Without it we OOM
    # at SFT because the activations for an 80M-trainable-param model on
    # ~500-token sequences exceed 24GB.
    if cfg["train"].get("grad_checkpointing", True):
        base = getattr(model, "base_model", model)
        base = getattr(base, "model", base)
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()
            base.config.use_cache = False
            text_emb = base.language_model.get_input_embeddings()
            def _require_grad_hook(_m, _i, o): o.requires_grad_(True)
            text_emb.register_forward_hook(_require_grad_hook)
            logger.info("Enabled gradient checkpointing + input-embeds grad hook.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    logger.info("Trainable params: %d", n_train)

    optimizer = torch.optim.AdamW(
        trainable, lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    # ---- Train loop ----
    rng = np.random.default_rng(cfg["data"].get("seed", 42))
    num_steps = int(cfg["train"]["num_steps"])
    batch_prompts = int(cfg["train"]["batch_prompts"])
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"sft_{timestamp}.jsonl"
    logger.info("Writing per-step log to %s", log_path)
    log_f = open(log_path, "a")

    model.train()
    losses_recent = []
    t_start = time.time()
    for step in range(num_steps):
        t0 = time.time()
        idx = rng.choice(len(sft_pool), size=batch_prompts, replace=False)
        batch = [sft_pool[int(i)] for i in idx]
        try:
            loss = sft_step(model, processor, batch, cfg, images_root, device)
        except FileNotFoundError as e:
            logger.warning("Skipping batch (file missing): %s", e); continue
        except torch.cuda.OutOfMemoryError as e:
            logger.error("OOM on step %d; skipping. %s", step, e)
            torch.cuda.empty_cache(); continue

        optimizer.zero_grad()
        loss.backward()
        # NaN safety (same as RL loop)
        n_nan = 0
        for p in trainable:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                n_nan += 1
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if n_nan:
            logger.warning("step %d: zeroed NaN/Inf in %d/%d grads",
                           step, n_nan, len(trainable))
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()

        loss_val = float(loss.detach())
        losses_recent.append(loss_val)
        dt = time.time() - t0
        log = {"step": step, "loss": loss_val, "time_s": dt}
        log_f.write(json.dumps(log) + "\n"); log_f.flush()

        if step % 10 == 0 or step == num_steps - 1:
            logger.info(
                "step=%d  loss=%.4f  (avg last 10 = %.4f)  dt=%.1fs",
                step, loss_val, float(np.mean(losses_recent[-10:])), dt,
            )

        if (step + 1) % cfg["checkpoint"]["save_every_n_steps"] == 0:
            save_dir = ckpt_dir / f"step_{step:06d}"
            model.save_pretrained(save_dir)
            if cfg.get("vision_swap"):
                base = getattr(model, "base_model", model)
                base = getattr(base, "model", base)
                torch.save(base.vision_tower.state_dict(),
                           save_dir / "vision_tower.pt")
            logger.info("Saved adapter to %s", save_dir)
            keep = cfg["checkpoint"].get("keep_last_n", 5)
            ckpts = sorted(ckpt_dir.glob("step_*"))
            for old in ckpts[:-keep]:
                logger.info("Removing old checkpoint %s", old)
                for f in old.iterdir(): f.unlink()
                old.rmdir()

    logger.info("Done. Total wall time %.1f min.", (time.time() - t_start) / 60)
    log_f.close()


if __name__ == "__main__":
    main()
