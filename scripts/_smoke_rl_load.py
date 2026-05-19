"""One-off smoke test: load MAIRA-2 + LoRA, sample one rollout on one prompt.

Verifies model loading, LoRA wrapping, image-input formatting, and sampled
generation before kicking off real training. Does NOT call the judge or do
a gradient step.

    python scripts/_smoke_rl_load.py
"""
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rl_maira2_temporal import (
    build_inputs, load_model_and_processor, sample_completions, to_device,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main():
    with open("configs/maira2_grpo_lora.yaml") as f:
        cfg = yaml.safe_load(f)
    # Use the less-busy GPU for the smoke test
    cfg["model"]["device"] = "cuda:1"
    cfg["train"]["grad_checkpointing"] = False  # no grad pass in smoke

    with open("data/rl_prompts/seed42/train.json") as f:
        pool = json.load(f)
    pair_prompt = pool["pair"][0]
    single_prompt = pool["single"][0]
    images_root = Path(cfg["data"]["images_root"])

    print("Loading model + LoRA ...")
    model, processor = load_model_and_processor(cfg)
    model.eval()

    for tag, p in (("PAIR", pair_prompt), ("SINGLE", single_prompt)):
        print(f"\n=== {tag} prompt: {p['prompt_id']} ===")
        print(f"finding={p['finding']}  gt={p['gt_progression']}  prior={p['prior_file']}")
        inputs = to_device(build_inputs(processor, p, cfg, images_root),
                            cfg["model"]["device"])
        print("inputs:", {k: tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__
                          for k, v in inputs.items()})
        cids, texts = sample_completions(
            model, processor, inputs, n=2,
            max_new_tokens=120, temperature=0.9, top_p=0.95,
        )
        print(f"completion_ids: {tuple(cids.shape)}")
        for i, t in enumerate(texts):
            print(f"--- completion {i} ---")
            print(t[:400])

    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
