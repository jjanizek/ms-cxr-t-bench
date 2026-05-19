# MAIRA-2 Temporal RL on Chest ImaGenome — Run 1 Results

GRPO + LoRA post-training of MAIRA-2 on Chest ImaGenome silver temporal labels (~64k pair-finding rows), with an LLM judge as verifier. Goal: improve MAIRA-2's downstream temporal-comparison accuracy on MS-CXR-T (held-out human-labeled set).

## Setup
- Model: `microsoft/maira-2` (~7B; Vicuna-7B LM + rad-DINO vision + 4-layer projector).
- Training data: 64,580 `(current, prior, finding, gt_progression)` rows from Chest ImaGenome silver labels (5 findings, subject-level filtered).
- Reward: 4-class LLM judge (`improving/stable/worsening/none`) on each sampled MAIRA-2 report; rewards `+1 / -0.5 / -1` (correct / adjacent class / opposite) under `partial_credit=true`.
- Algorithm: GRPO with `G=4` rollouts/prompt, `B=2` prompts/batch, KL to base (β=0.1), lr 5e-6, 1000 steps.
- LoRA targets: LM attention + MLP (r=16, ~40M trainable). Vision LoRA disabled (PEFT/Llava grad-flow issue — to be debugged separately).
- Eval every 100 steps on 100 held-out ImaGenome val pairs.

## Headline numbers (MS-CXR-T val+test, n=315, OG 3-class judge, macro_acc per finding then averaged)

| Model | avg macro_acc | exact-match pair_acc |
|---|---:|---:|
| Baseline MAIRA-2 | 0.325 | 0.406 |
| ImaGenome RL step_99 | 0.344 | 0.422 |
| ImaGenome RL step_699 | **0.352** | 0.432 |
| ImaGenome RL step_999 (final) | 0.351 | **0.448** |

Per-finding (baseline → step_999):

| Finding | Baseline | step_699 | step_999 |
|---|---:|---:|---:|
| consolidation | 0.335 | 0.364 | 0.336 |
| edema | 0.391 | 0.386 | 0.333 |
| pleural_effusion | 0.341 | 0.304 | 0.344 |
| pneumonia | 0.222 | 0.299 | 0.335 |
| pneumothorax | 0.336 | 0.409 | 0.407 |

For published context: BioViL-T full-FT on ImaGenome reaches **0.612** macro_acc on the same MS-CXR-T task. We are nowhere near that yet.

## Subgroup analysis — critical finding

Per-gt-class accuracy on the same 315 pairs:

| GT class | n | baseline | step_699 | step_999 |
|---|---:|---:|---:|---:|
| improving | 57 | 0.035 | 0.140 | **0.000** |
| stable | 137 | 0.737 | 0.818 | **0.985** |
| worsening | 121 | 0.207 | 0.132 | **0.050** |

| | baseline | step_699 | step_999 |
|---|---:|---:|---:|
| accuracy on real changes (improving + worsening) | 0.152 | 0.135 | **0.034** |
| accuracy on stable cases | 0.737 | 0.818 | **0.985** |

**The model collapsed to predicting "stable" 97% of the time.** On the 178 real-change pairs:
- Baseline: 27 correct, 135 hedged-to-stable, 16 wrong-direction
- step_999: **6 correct**, 172 hedged-to-stable, 0 wrong-direction

The macro_acc gain (0.325 → 0.351) is an artifact: stable's per-class recall jumped to 0.985, inflating the macro-averaged metric while accuracy on the clinically important "real change" cases regressed by 4×.

## Why this happened — reward shape

With `partial_credit=true`:
- Predicting `stable` when gt is `improving/worsening`: reward = −0.5 (adjacent class)
- Predicting wrong direction (`improving` when gt = `worsening`): reward = −1.0
- Predicting correctly: reward = +1.0

Combined with the 43% stable prior in the gt distribution, the policy discovered that "always predict stable" is the lowest-variance, lowest-expected-loss strategy. KL drift was kept in check (~0.10–0.20 throughout) but evidently not against the mode-collapse direction.

## Training trajectory (held-out ImaGenome val, n=100, 4-class judge)

| step | greedy | sampled | KL | R_lbl trend |
|---:|---:|---:|---:|---:|
| 99 | 0.18 | 0.11 | 0.000 | -0.5 |
| 199 | 0.17 | 0.21 | 0.001 | -0.4 |
| 299 | 0.28 | 0.32 | 0.002 | -0.4 |
| 399 | 0.31 | 0.29 | 0.004 | -0.3 |
| 499 | 0.30 | 0.33 | 0.008 | -0.3 |
| 599 | 0.36 | 0.28 | 0.015 | -0.3 |
| 699 | **0.46** | 0.39 | 0.025 | -0.3 |
| 799 | 0.38 | 0.41 | 0.060 | -0.2 |
| 899 | 0.41 | **0.44** | 0.090 | -0.2 |
| 999 | 0.40 | 0.41 | 0.150 | -0.2 |

The trajectory looked like real learning until ~step 600, then the policy started exhibiting larger KL spikes (up to KL≈3) and the gains went into mode-collapse rather than improved discrimination.

## Example reports (MS-CXR-T pair, finding=consolidation, gt=worsening)

> **Baseline → worsening ✓**
> *"... A right mid lung opacity is seen, which is more conspicuous compared to the prior exam. ..."*

> **step_699 → stable ✗**
> *"Since the prior study, there has been no significant change in the right mid to lower lung consolidation. ..."*

> **step_999 → stable ✗**
> *"Since the prior study, there has been no significant change in the right lower lung consolidation. ..."*

The trained checkpoints learned a fluent comparison template ("Since the prior study, there has been no significant change…") but converged on this stable-leaning template even when the actual finding had changed.

## Lessons for next run

1. **Drop `partial_credit`** — restore the binary reward (+1 / −1). Removes the asymmetric incentive that made "stable" the low-risk hedge.
2. **Stratify batches by gt class as well as by finding** — currently batches can be 60%+ stable, mirroring the prior; this biases the gradient direction.
3. **Track per-gt-class accuracy in the in-training eval** — would have surfaced this collapse at step 300 instead of step 999.
4. **Asymmetric anti-hedging reward** (more aggressive option) — penalize "stable when changing" *more* than "wrong direction" to actively reward commitment.
5. **Smaller `kl_coef` may not be the answer** — the policy drifted moderately and still collapsed. The reward shape was the problem.

## Files
- `scripts/rl_maira2_temporal.py` — GRPO+LoRA training loop
- `scripts/build_imagenome_rl_prompts.py` — pair-prompt pool builder for ImaGenome
- `configs/maira2_grpo_imagenome.yaml` — config for this run
- `evaluation/eval_og_metric.py` — apples-to-apples OG-judge eval
- `scripts/plot_rl_curves.py` — training curve diagnostics
- Checkpoints under `checkpoints/maira2_grpo_imagenome/step_{599,699,799,899,999}` (last 5 retained)
- Predictions and summaries under `results/maira2_og_metric_eval/`
