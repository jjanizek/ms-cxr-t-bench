"""GRPO + LoRA post-training of MAIRA-2 for temporal CXR comparison.

Reward = α · R_label + β · R_no_compare, where:
  - pair prompts:    R_label = +1 if judge classification matches gt, else -1.
                     R_no_compare = 0.
  - single prompts:  R_label = 0.
                     R_no_compare = +1 if judge.makes_comparison == False else -1
                     (single-image input → no prior → any comparative claim is
                      a hallucinated prior, in the spirit of Ramesh et al. 2024).

Architecture: LoRA adapters on language_model.* attention + MLP projections.
Vision tower (DINOv2) and multimodal projector are frozen. Reference policy
for KL penalty is the same model with adapters disabled (PEFT
`model.disable_adapter()`), so no second copy in GPU memory.

GRPO advantages are computed per (prompt, group of G rollouts) with
within-group normalization. KL penalty uses Schulman's k_2 estimator
(unbiased, non-negative).

Usage:
    python scripts/rl_maira2_temporal.py --config configs/maira2_grpo_lora.yaml
"""
import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Silence noisy 3rd-party loggers
for noisy in ("httpx", "openai._base_client", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# The checkpoint warning fires harmlessly for the ref-model forward (no_grad
# context where grad propagation is intentionally disabled).
import warnings as _w
_w.filterwarnings("ignore",
                  message=r".*None of the inputs have requires_grad=True.*")


FINDING_DISPLAY = {
    "consolidation": "consolidation",
    "edema": "pulmonary edema",
    "pleural_effusion": "pleural effusion",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
}

LABEL_MAP = {"improving": 0, "stable": 1, "worsening": 2, "none": 3}


# ============================================================
# Judge (async)
# ============================================================

JUDGE_SYSTEM_PROMPT = """\
You are an expert radiologist. You will be given a radiology report describing \
chest X-ray findings, and a specific clinical finding to evaluate. Extract two \
things from the report:

1. CLASSIFICATION of the finding's temporal progression:
   - "improving": described as getting better, resolving, decreasing, or improved
   - "stable": described as unchanged, similar, stable, or persistent without change
   - "worsening": described as getting worse, increasing, new, progressing, or worsened
   - "none": the report does NOT commit to a specific progression for this finding \
(either does not mention it, or mentions it only descriptively with no temporal claim)

2. MAKES_COMPARISON: whether the report makes ANY comparative claim about this \
finding (or its absence) relative to a prior study. Comparative phrases include \
"increased", "decreased", "new", "improved", "worsened", "compared to prior", \
"interval change", "unchanged", "stable", "persistent", "previously seen", etc. \
Return true if any such comparative language is used about this finding; false if \
the report describes the finding only in static terms (e.g., "small left pleural \
effusion" with no temporal qualifier) or does not mention the finding at all.

Note: "stable" and "unchanged" ARE comparative claims (they assert no change \
relative to prior). A purely descriptive statement with no temporal framing is NOT.

Respond with EXACTLY this JSON (no other text):
{"classification": "improving" | "stable" | "worsening" | "none", \
"makes_comparison": true | false, \
"reasoning": "one sentence explanation"}
"""

JUDGE_USER_TEMPLATE = """\
Report:
{report}

Finding to evaluate: {finding}

Classify the temporal progression of {finding} as improving, stable, worsening, \
or none, and indicate whether the report makes any comparative claim."""


def parse_judge_response(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
        cls = parsed.get("classification", "").lower().strip()
        mc = parsed.get("makes_comparison", None)
        if isinstance(mc, str):
            mc = mc.strip().lower() in ("true", "yes", "1")
        if cls in LABEL_MAP and isinstance(mc, bool):
            return {"classification": cls, "makes_comparison": mc,
                    "reasoning": parsed.get("reasoning", "")}
    except json.JSONDecodeError:
        pass
    return {"classification": "none", "makes_comparison": False,
            "reasoning": f"PARSE_FAILED: {text[:200]}"}


class AsyncJudge:
    """Async OpenAI judge with bounded concurrency.

    The Semaphore is created per grade_many call rather than once at __init__,
    because asyncio.Semaphore binds to the current event loop on first use and
    `asyncio.run` creates a new loop each invocation — reusing the semaphore
    across asyncio.run calls raises "bound to a different event loop".
    """
    def __init__(self, model: str, concurrency: int = 20):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
        self.model = model
        self.concurrency = concurrency

    async def grade_one(self, sem: asyncio.Semaphore,
                         report: str, finding: str) -> dict:
        display = FINDING_DISPLAY.get(finding, finding)
        user = JUDGE_USER_TEMPLATE.format(report=report, finding=display)
        async with sem:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                              {"role": "user", "content": user}],
                    temperature=0, max_tokens=150,
                )
                return parse_judge_response(resp.choices[0].message.content)
            except Exception as e:
                logger.warning("Judge API error: %s", e)
                return {"classification": "none", "makes_comparison": False,
                        "reasoning": f"API_ERROR: {e}"}

    async def grade_many(self, items: list[tuple[str, str]]) -> list[dict]:
        sem = asyncio.Semaphore(self.concurrency)
        return await asyncio.gather(*(self.grade_one(sem, r, f) for r, f in items))


# ============================================================
# Reward
# ============================================================

def compute_reward(prompt_type: str, gt: str | None,
                   judge_out: dict, alpha: float, beta: float,
                   partial_credit: bool = False) -> tuple[float, dict]:
    cls = judge_out["classification"]
    mc = judge_out["makes_comparison"]

    if prompt_type == "pair":
        if cls == gt:
            r_label = 1.0
        elif partial_credit:
            order = ["improving", "stable", "worsening"]
            if cls in order and gt in order and abs(order.index(cls) - order.index(gt)) == 1:
                r_label = -0.5
            else:
                r_label = -1.0
        else:
            r_label = -1.0
        r_no_compare = 0.0
    else:  # single
        r_label = 0.0
        r_no_compare = 1.0 if mc is False else -1.0

    reward = alpha * r_label + beta * r_no_compare
    return reward, {"r_label": r_label, "r_no_compare": r_no_compare,
                    "classification": cls, "makes_comparison": mc}


# ============================================================
# Model loading + LoRA wrapping
# ============================================================

def load_model_and_processor(cfg: dict):
    from transformers import AutoModelForCausalLM, AutoProcessor
    from peft import LoraConfig, get_peft_model

    model_name = cfg["model"]["name"]
    dtype = getattr(torch, cfg["model"]["torch_dtype"])
    device = cfg["model"]["device"]

    logger.info("Loading processor + model from %s ...", model_name)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=dtype, device_map={"": device},
    )

    # Freeze everything; PEFT will unfreeze only LoRA params.
    for p in model.parameters():
        p.requires_grad = False

    target_modules = list(cfg["lora"]["target_modules"])

    # Optionally apply LoRA to the last N DINOv2 vision-tower layers as well.
    # PEFT's target_modules list matches by str.endswith, so passing the FULL
    # module path for a specific layer restricts the adapter to exactly that
    # layer (vs. just "query" which would match all 12 layers).
    vcfg = cfg.get("vision_lora", {}) or {}
    if vcfg.get("enabled"):
        n_layers_total = 12  # MAIRA-2 uses 12-layer rad-DINO
        last_n = int(vcfg.get("last_n_layers", 6))
        first = max(0, n_layers_total - last_n)
        added = []
        for li in range(first, n_layers_total):
            for proj in ("query", "key", "value"):
                added.append(f"vision_tower.encoder.layer.{li}.attention.attention.{proj}")
            added.append(f"vision_tower.encoder.layer.{li}.attention.output.dense")
        target_modules.extend(added)
        logger.info("Vision LoRA enabled on layers %d–%d (%d modules added).",
                    first, n_layers_total - 1, len(added))

    lora = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    if cfg["train"].get("grad_checkpointing"):
        base = getattr(model, "base_model", model)
        base = getattr(base, "model", base)  # Maira2ForConditionalGeneration
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()
            base.config.use_cache = False
            # PEFT's enable_input_require_grads hooks model.get_input_embeddings(),
            # but for multimodal Llava the LM consumes a merged (text+image) embedding
            # tensor — we need the *text* embedding output to require grad so the
            # merged tensor (and thus checkpointed layers) carries grad.
            text_emb = base.language_model.get_input_embeddings()
            def _require_grad_hook(_m, _i, o):
                o.requires_grad_(True)
            text_emb.register_forward_hook(_require_grad_hook)
            logger.info("Enabled gradient checkpointing + input-embeds grad hook.")

    return model, processor


# ============================================================
# Prompt → inputs (images + chat template)
# ============================================================

def build_inputs(processor, prompt: dict, cfg: dict, images_root: Path):
    curr = Image.open(images_root / prompt["current_file"]).convert("RGB")
    prior = None
    if prompt["prior_file"] is not None:
        prior = Image.open(images_root / prompt["prior_file"]).convert("RGB")

    if cfg["prompting"]["prompt_mode"] == "specific":
        display = FINDING_DISPLAY.get(prompt["finding"], prompt["finding"])
        comparison = (
            f"{'Prior study available. ' if prior is not None else ''}"
            f"Describe the current chest X-ray and "
            f"{'any change in ' + display if prior is not None else 'any findings of ' + display}."
        )
    else:
        comparison = "Prior study available." if prior is not None else None

    return processor.format_and_preprocess_reporting_input(
        current_frontal=curr,
        current_lateral=None,
        prior_frontal=prior,
        indication=cfg["prompting"].get("indication", "") or None,
        technique=cfg["prompting"].get("technique") or None,
        comparison=comparison,
        prior_report=None,
        return_tensors="pt",
        get_grounding=False,
    )


def to_device(batch, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


# ============================================================
# Rollout sampling
# ============================================================

@torch.no_grad()
def sample_completions(model, processor, inputs, n: int,
                        max_new_tokens: int, temperature: float, top_p: float):
    """Return (completion_ids [n, T], completion_texts [n]).

    We use no_grad (not inference_mode) so the returned completion_ids can be
    used as gather indices in compute_logprobs without "inference tensors
    cannot be saved for backward" errors.
    """
    out = model.generate(
        **inputs,
        do_sample=True,
        num_return_sequences=n,
        max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[-1]
    completion_ids = out[:, prompt_len:].clone()
    texts = []
    for i in range(n):
        decoded = processor.decode(completion_ids[i], skip_special_tokens=True).lstrip()
        texts.append(_safe_parse(processor, decoded))
    return completion_ids, texts


def _safe_parse(processor, decoded: str) -> str:
    """Convert MAIRA-2 output to plain text, falling back gracefully on
    malformed grounding tokens (RL can push outputs off-manifold)."""
    try:
        parsed = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
        return (parsed if isinstance(parsed, str) else str(parsed)).strip()
    except (AssertionError, ValueError, IndexError):
        # Strip MAIRA-2 grounding tokens defensively
        for tok in ("<obj>", "</obj>", "<box>", "</box>"):
            decoded = decoded.replace(tok, "")
        return decoded.strip()


# ============================================================
# Per-token log-prob computation
# ============================================================

def _compute_logprobs_chunk(model, processor, inputs, completion_ids, use_adapter):
    n, T_c = completion_ids.shape
    prompt_ids = inputs["input_ids"]
    prompt_attn = inputs["attention_mask"]

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    prompt_ids_n = prompt_ids.expand(n, -1)
    prompt_attn_n = prompt_attn.expand(n, -1)
    completion_mask = (completion_ids != pad_id).long()
    full_ids = torch.cat([prompt_ids_n, completion_ids], dim=1)
    full_attn = torch.cat([prompt_attn_n, completion_mask], dim=1)

    image_kwargs = {}
    for k in ("pixel_values", "image_sizes"):
        if k in inputs:
            v = inputs[k]
            repeat_dims = [n] + [1] * (v.dim() - 1)
            image_kwargs[k] = v.repeat(*repeat_dims)

    ctx = (torch.no_grad() if not use_adapter else torch.enable_grad())
    adapter_ctx = (model.disable_adapter() if not use_adapter else _NullContext())
    with adapter_ctx, ctx:
        out = model(input_ids=full_ids, attention_mask=full_attn,
                    use_cache=False, **image_kwargs)
    logits = out.logits[:, -T_c - 1:-1, :]
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
    return tok_logp, completion_mask.float()


def compute_logprobs(model, processor, inputs: dict, completion_ids: torch.Tensor,
                     use_adapter: bool, chunk_size: int = 2):
    """Compute per-token log-probs of completion_ids given the prompt+images.

    Chunks over the n dimension to bound peak memory: a chunk of size c does
    one forward over c sequences, then concatenates results across chunks.
    Set chunk_size >= n to disable chunking.
    """
    n = completion_ids.shape[0]
    chunks = []
    masks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sub = completion_ids[start:end]
        lp, m = _compute_logprobs_chunk(model, processor, inputs, sub, use_adapter)
        chunks.append(lp); masks.append(m)
    return torch.cat(chunks, dim=0), torch.cat(masks, dim=0)


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ============================================================
# GRPO loss
# ============================================================

def grpo_loss_per_sample(policy_logp: torch.Tensor, ref_logp: torch.Tensor,
                          mask: torch.Tensor, advantages: torch.Tensor,
                          kl_coef: float, ppo_clip: float,
                          old_logp: torch.Tensor = None):
    """Return per-sample (loss, surrogate, kl) of shape [n].

    Caller decides aggregation so chunked policy forwards can backward each
    chunk independently (bounded peak memory) while still computing the
    correct mini-batch-mean loss across all chunks.
    """
    seq_logp_policy = (policy_logp * mask).sum(-1)            # [n]
    if old_logp is None:
        surrogate = advantages * seq_logp_policy
    else:
        seq_logp_old = (old_logp * mask).sum(-1).detach()
        ratio = torch.exp(seq_logp_policy - seq_logp_old)
        clipped = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip)
        surrogate = torch.min(ratio * advantages, clipped * advantages)

    r = (ref_logp - policy_logp) * mask
    kl_per_tok = torch.exp(r) - r - 1.0
    kl = (kl_per_tok * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)  # [n]

    return -surrogate + kl_coef * kl, surrogate.detach(), kl.detach()


# ============================================================
# Eval (greedy)
# ============================================================

@torch.inference_mode()
def quick_eval(model, processor, prompts: list[dict], cfg: dict,
               images_root: Path, judge: AsyncJudge,
               max_pair: int, max_single: int, device: str) -> dict:
    model.eval()
    rng = np.random.default_rng(0)
    pair = [p for p in prompts if p["prompt_type"] == "pair"]
    single = [p for p in prompts if p["prompt_type"] == "single"]
    rng.shuffle(pair); rng.shuffle(single)
    sample = pair[:max_pair] + single[:max_single]

    reports = []           # (prompt, greedy_report)
    sampled_reports = []   # (prompt, sampled_report) — pairs only
    for p in tqdm(sample, desc="eval", leave=False):
        inputs = to_device(build_inputs(processor, p, cfg, images_root), device)
        # Greedy
        out = model.generate(**inputs, do_sample=False,
                             max_new_tokens=cfg["sampling"]["max_new_tokens"],
                             use_cache=True,
                             pad_token_id=processor.tokenizer.pad_token_id
                                          or processor.tokenizer.eos_token_id)
        T_p = inputs["input_ids"].shape[-1]
        text = processor.decode(out[0][T_p:], skip_special_tokens=True).lstrip()
        reports.append((p, _safe_parse(processor, text)))

        # One sampled completion per pair prompt (for sampled-acc comparison)
        if p["prompt_type"] == "pair":
            out_s = model.generate(**inputs, do_sample=True,
                                    temperature=cfg["sampling"]["temperature"],
                                    top_p=cfg["sampling"]["top_p"],
                                    max_new_tokens=cfg["sampling"]["max_new_tokens"],
                                    use_cache=True,
                                    pad_token_id=processor.tokenizer.pad_token_id
                                                  or processor.tokenizer.eos_token_id)
            text_s = processor.decode(out_s[0][T_p:], skip_special_tokens=True).lstrip()
            sampled_reports.append((p, _safe_parse(processor, text_s)))

    judge_items = [(rep, p["finding"]) for p, rep in reports]
    judge_outs = asyncio.run(judge.grade_many(judge_items))
    sampled_judge_outs = []
    if sampled_reports:
        sampled_judge_outs = asyncio.run(judge.grade_many(
            [(rep, p["finding"]) for p, rep in sampled_reports]))

    pair_correct, pair_total = 0, 0
    single_no_comp, single_total = 0, 0
    for (p, _), jo in zip(reports, judge_outs):
        if p["prompt_type"] == "pair":
            pair_total += 1
            if jo["classification"] == p["gt_progression"]:
                pair_correct += 1
        else:
            single_total += 1
            if jo["makes_comparison"] is False:
                single_no_comp += 1

    sampled_pair_correct = sum(
        1 for (p, _), jo in zip(sampled_reports, sampled_judge_outs)
        if jo["classification"] == p["gt_progression"]
    )

    return {
        "pair_acc_greedy": pair_correct / max(pair_total, 1),
        "pair_acc_sampled": sampled_pair_correct / max(len(sampled_reports), 1),
        "pair_n": pair_total,
        "single_no_comparison_rate": single_no_comp / max(single_total, 1),
        "single_n": single_total,
    }


# ============================================================
# Main loop
# ============================================================

def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                        cwd=Path(__file__).parent.parent,
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="Path to LoRA adapter dir")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["logging"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"run_{run_id}.jsonl"

    device = cfg["model"]["device"]
    images_root = Path(cfg["data"]["images_root"])

    # Load prompts
    prompts_dir = Path(cfg["data"]["prompts_dir"])
    with open(prompts_dir / "train.json") as f:
        train_pool = json.load(f)
    with open(prompts_dir / "val.json") as f:
        val_pool = json.load(f)
    train_pair = train_pool["pair"]
    train_single = train_pool["single"]
    val_all = val_pool["pair"] + val_pool["single"]
    logger.info("Train: pair=%d single=%d  | Val: pair=%d single=%d",
                len(train_pair), len(train_single),
                len(val_pool["pair"]), len(val_pool["single"]))

    # Model
    model, processor = load_model_and_processor(cfg)
    if args.resume:
        from peft import PeftModel
        model.load_adapter(args.resume, adapter_name="default")
        logger.info("Resumed LoRA adapter from %s", args.resume)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"],
                                   weight_decay=cfg["train"]["weight_decay"])

    judge = AsyncJudge(model=cfg["judge"]["model"],
                       concurrency=cfg["judge"]["concurrency"])

    rng = np.random.default_rng(0)
    B = cfg["train"]["batch_prompts"]
    G = cfg["sampling"]["group_size"]
    ratio = cfg["data"]["pair_single_ratio"]
    kl_coef = cfg["train"]["kl_coef"]
    grad_clip = cfg["train"]["grad_clip"]
    alpha = cfg["reward"]["alpha_label"]
    beta = cfg["reward"]["beta_no_compare"]
    partial = cfg["reward"]["partial_credit"]
    stratify_by_finding = bool(cfg["data"].get("stratify_by_finding", False))

    # Per-prompt sampling weights — when stratify_by_finding is on, weight each
    # prompt by 1/finding_count so every finding has equal expected airtime
    # regardless of pool size (ImaGenome pleural_effusion has ~6× consolidation).
    def _weights(pool):
        if not pool or not stratify_by_finding:
            return None
        from collections import Counter
        counts = Counter(p["finding"] for p in pool)
        w = np.array([1.0 / counts[p["finding"]] for p in pool], dtype=np.float64)
        return w / w.sum()
    pair_weights = _weights(train_pair)
    single_weights = _weights(train_single)

    def _draw(pool, n, weights):
        if n == 0 or not pool:
            return []
        idx = rng.choice(len(pool), size=min(n, len(pool)),
                          replace=False, p=weights)
        return [pool[i] for i in idx]

    for step in range(cfg["train"]["num_steps"]):
        t0 = time.time()
        # Sample batch: B prompts, mix pair/single
        n_pair = int(round(B * ratio))
        n_single = B - n_pair
        batch_prompts = (
            _draw(train_pair, n_pair, pair_weights) +
            _draw(train_single, n_single, single_weights)
        )

        # Rollouts (no grad)
        model.eval()
        rollouts = []
        for p in batch_prompts:
            inputs = to_device(build_inputs(processor, p, cfg, images_root), device)
            completion_ids, texts = sample_completions(
                model, processor, inputs, n=G,
                max_new_tokens=cfg["sampling"]["max_new_tokens"],
                temperature=cfg["sampling"]["temperature"],
                top_p=cfg["sampling"]["top_p"],
            )
            rollouts.append({"prompt": p, "inputs": inputs,
                             "completion_ids": completion_ids, "texts": texts})

        # Release KV-cache memory held by generate() before the gradient pass
        # (G=8 sampling can leave several GB of fragmented cache that triggers
        # OOM on the subsequent forward+backward).
        torch.cuda.empty_cache()

        # Judge (async, batched across all rollouts in step)
        judge_items = [(t, r["prompt"]["finding"])
                       for r in rollouts for t in r["texts"]]
        judge_outs_flat = asyncio.run(judge.grade_many(judge_items))
        # Re-split by rollout
        cursor = 0
        for r in rollouts:
            r["judge"] = judge_outs_flat[cursor:cursor + G]
            cursor += G

        # Rewards + advantages
        for r in rollouts:
            rewards = []
            comps = []
            for jo in r["judge"]:
                rew, comp = compute_reward(
                    r["prompt"]["prompt_type"], r["prompt"]["gt_progression"],
                    jo, alpha, beta, partial,
                )
                rewards.append(rew); comps.append(comp)
            r["rewards"] = np.array(rewards, dtype=np.float32)
            r["reward_parts"] = comps
            mu, sd = r["rewards"].mean(), r["rewards"].std()
            r["advantages"] = (r["rewards"] - mu) / (sd + 1e-6)

        # Gradient pass — per-chunk forward+backward so peak activations stay
        # bounded at chunk_size sequences (without this, chunking accumulates
        # graphs across chunks and OOMs at G=8 just like un-chunked G=8 would).
        model.train()
        optimizer.zero_grad()
        chunk = int(cfg["train"].get("logp_chunk_size", 2))
        ppo_clip = float(cfg["train"].get("ppo_clip", 0.2))
        total_samples = sum(r["completion_ids"].shape[0] for r in rollouts)
        step_loss_sum = 0.0
        step_surrogate_sum = 0.0
        step_kl_sum = 0.0
        for r in rollouts:
            inputs = r["inputs"]
            cids = r["completion_ids"]
            adv_full = torch.tensor(r["advantages"], device=device, dtype=torch.float32)

            # Reference logprobs (adapter off, no grad). Chunked but no
            # graph retention — torch.no_grad inside the helper context.
            was_training = model.training
            model.eval()
            with torch.no_grad():
                ref_logp_full, mask_full = compute_logprobs(
                    model, processor, inputs, cids,
                    use_adapter=False, chunk_size=chunk,
                )
            ref_logp_full = ref_logp_full.detach()
            mask_full = mask_full.detach()
            if was_training:
                model.train()

            # Policy: per-chunk forward+backward, accumulate gradients.
            n = cids.shape[0]
            for cstart in range(0, n, chunk):
                cend = min(cstart + chunk, n)
                cids_c = cids[cstart:cend]
                ref_c = ref_logp_full[cstart:cend]
                mask_c = mask_full[cstart:cend]
                adv_c = adv_full[cstart:cend]

                pol_lp_c, _ = _compute_logprobs_chunk(
                    model, processor, inputs, cids_c, use_adapter=True,
                )
                per_sample_loss, surr, kl = grpo_loss_per_sample(
                    pol_lp_c, ref_c, mask_c, adv_c,
                    kl_coef=kl_coef, ppo_clip=ppo_clip,
                )
                # Scale so the sum across all chunks across all rollouts
                # equals the mean per-sample loss for this step.
                chunk_loss = per_sample_loss.sum() / total_samples
                chunk_loss.backward()

                step_loss_sum += float(per_sample_loss.detach().sum())
                step_surrogate_sum += float(surr.sum())
                step_kl_sum += float(kl.sum())

        step_loss = step_loss_sum / max(total_samples, 1)
        step_surrogate = step_surrogate_sum / max(total_samples, 1)
        step_kl = step_kl_sum / max(total_samples, 1)

        n_with_grad = sum(1 for p in trainable if p.grad is not None)
        gn = sum(float(p.grad.abs().sum()) for p in trainable if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()

        dt = time.time() - t0
        # Aggregate reward stats
        all_rewards = np.concatenate([r["rewards"] for r in rollouts])
        all_r_label = np.array([c["r_label"] for r in rollouts for c in r["reward_parts"]])
        all_r_nc = np.array([c["r_no_compare"] for r in rollouts for c in r["reward_parts"]])
        log = {
            "step": step,
            "time_s": round(dt, 1),
            "loss": step_loss,
            "surrogate": step_surrogate,
            "kl": step_kl,
            "reward_mean": float(all_rewards.mean()),
            "reward_std": float(all_rewards.std()),
            "r_label_mean": float(all_r_label.mean()),
            "r_no_compare_mean": float(all_r_nc.mean()),
            "n_pair": n_pair, "n_single": n_single,
            "params_with_grad": n_with_grad, "grad_l1": gn,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log) + "\n")
        if step % cfg["logging"]["log_every_n_steps"] == 0:
            logger.info(
                "step=%d  loss=%.3f  kl=%.3f  R=%.2f±%.2f  R_lbl=%.2f  R_nc=%.2f  "
                "grad_l1=%.4f  pwg=%d/%d  (%.1fs)",
                step, log["loss"], log["kl"], log["reward_mean"], log["reward_std"],
                log["r_label_mean"], log["r_no_compare_mean"],
                gn, n_with_grad, len(trainable), dt,
            )

        if (step + 1) % cfg["eval"]["every_n_steps"] == 0:
            # Save adapter FIRST so eval crashes don't lose training progress.
            save_dir = ckpt_dir / f"step_{step:06d}"
            model.save_pretrained(save_dir)
            logger.info("Saved adapter to %s", save_dir)

            # Trim old checkpoints
            keep = cfg["checkpoint"]["keep_last_n"]
            ckpts = sorted(ckpt_dir.glob("step_*"))
            for old in ckpts[:-keep]:
                logger.info("Removing old checkpoint %s", old)
                for f in old.iterdir():
                    f.unlink()
                old.rmdir()

            try:
                metrics = quick_eval(
                    model, processor, val_all, cfg, images_root, judge,
                    max_pair=cfg["eval"]["n_pair_eval"],
                    max_single=cfg["eval"]["n_single_eval"],
                    device=device,
                )
                metrics["step"] = step
                logger.info(
                    "EVAL step=%d  pair_acc_greedy=%.3f  pair_acc_sampled=%.3f  "
                    "single_no_comp=%.3f",
                    step, metrics["pair_acc_greedy"], metrics["pair_acc_sampled"],
                    metrics["single_no_comparison_rate"],
                )
                with open(out_dir / f"eval_{run_id}.jsonl", "a") as f:
                    f.write(json.dumps(metrics) + "\n")
                with open(save_dir / "meta.json", "w") as f:
                    json.dump({"step": step, "metrics": metrics,
                               "git_hash": git_hash(), "config": cfg}, f, indent=2)
            except Exception as e:
                logger.exception("EVAL step=%d failed: %s — continuing training", step, e)
                with open(save_dir / "meta.json", "w") as f:
                    json.dump({"step": step, "metrics": None, "eval_error": str(e),
                               "git_hash": git_hash(), "config": cfg}, f, indent=2)


if __name__ == "__main__":
    main()
