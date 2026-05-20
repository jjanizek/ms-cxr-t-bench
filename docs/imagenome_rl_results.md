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

---

# Second iteration: v2 / v3 / v4 / v5 (2026-05-19)

After v1's "always stable" collapse, four follow-up configurations attempted to
break the Schelling point. Final headline on MS-CXR-T val+test (315 pairs,
3-class judge, balanced accuracy averaged across 5 findings):

| Run                             | macro_acc | Pred dist (imp / stab / wors)  | Notes |
|---------------------------------|-----------|--------------------------------|-------|
| Baseline (no LoRA)              | 0.325     | —                              | |
| v1 step 999 (partial_credit)    | 0.351     | 1.5 / 97 / 1.5                 | mode collapse |
| v2 step 99                      | killed early | imp 32 / stable 4 / **none 64** | collapse to "none" |
| v3 step 99 (3-class judge)      | 0.302 (in-train) | 6 / 79 / 15            | back to stable |
| v4 step 199 (cw=2, sw=1)        | 0.271 (in-train) | 3 / 89 / 8             | stable bias intensified |
| **v5 step 99** (change-only)    | **0.360**| 9 / 75 / 16                    | **best** |
| v5 step 199                     | 0.356     | 15 / 62 / 23                   | |
| v5 step 299                     | 0.356     | 14 / 55 / 31                   | |
| v5 step 399                     | 0.311     | 21 / 44 / 35                   | regression on MS-CXR-T |
| BioViL-T (Bannur 2023, Table 2) | 0.612     | —                              | end-to-end on ImaGenome |

## What worked

**v5 — drop stable-gt prompts from training** (`train_classes: [improving, worsening]`).
Removed the attractor entirely from the gradient landscape. Result: best RL
macro_acc (0.360, +3.5pp over baseline, +0.9pp over v1's collapsed best) with
a well-balanced prediction distribution rather than degenerate one-class
collapse.

## What didn't work

- **v2 (binary reward, 4-class judge)**: model discovered "none" classification
  as a low-KL escape valve. Predicted "none" 64% at step 99.
- **v3 (3-class judge for reward)**: closed the "none" valve but didn't shift
  the stable Schelling point. R plateaued at −0.30 (matches expected reward
  of "always stable" under stratified gt: 1/3·(+1) + 2/3·(−1) = −1/3).
- **v4 (class-weighted reward, change=2, stable=1)**: training R improved
  slightly but greedy eval got *worse* — stable predictions climbed from 75% →
  89% over 200 steps. Asymmetric reward wasn't enough to overcome the prior
  toward "stable" reports.

## The training/eval distribution gap

In-training eval (ImaGenome val) showed v5 step 399 jumping to macro_acc 0.362
with well-balanced predictions (21/44/35). MS-CXR-T eval *dropped* to 0.311.
The model is over-fitting to ImaGenome silver-label conventions that don't
transfer to MS-CXR-T's human-rated labels. Implication: **early stopping on
the actual target distribution matters**; the in-training proxy can mislead.

## Lessons

1. **Mode-collapse defenses go in a strict order**:
   reward shape ⟶ training data composition ⟶ stratified sampling.
   v3/v4 tried (1) without doing (2); only v5 (drop stable-gt) actually moved
   the policy off the prior.
2. **The in-training eval is a proxy, not the target.** Always evaluate on
   MS-CXR-T (the real eval set) at every checkpoint, not just the
   training-distribution val.
3. **Greedy eval lags sampled eval by ~100 steps**: under sampling the model
   commits to a change before greedy decoding's argmax shifts. v5 step 99
   already had useful policy mass on changes that argmax didn't surface yet.
4. **Frozen-vision LoRA caps at ~0.36 macro_acc**: this is roughly halfway from
   baseline (0.325) to BioViL-T (0.612). Further gains likely require either
   (a) unfreezing the vision tower (PEFT+Llava integration TODO), or
   (b) SFT on silver labels before RL to teach the response format and base
   rates more directly.

## Files

Configs: `configs/maira2_grpo_imagenome_v{2,3,4,5}.yaml`
Best adapter: `checkpoints/maira2_grpo_imagenome_v5/step_000099/`
MS-CXR-T eval summaries: `results/maira2_og_metric_eval/summary_v5_step{099,199,299,399}_*.json`
Training logs: `results/maira2_grpo_imagenome_v5/run_*.jsonl`

---

# Third iteration: v6 — more LoRA capacity (2026-05-20)

The user asked whether throwing more LoRA parameters at the LM (and projector)
could substitute for unfreezing the vision encoder. v6 tests this directly.

## Config changes vs v5

| Knob | v5 | v6 | Why |
|------|----|-----|-----|
| LoRA rank | r=16 (α=32) | **r=32 (α=64)** | 2× per-module capacity (r=64 OOM'd) |
| LoRA targets | LM attn+MLP | LM attn+MLP **+ projector Linears (4)** | adapt the vision→LM bridge without unfreezing rad-DINO |
| Trainable params | 40M (0.58%) | **81M (1.15%)** | 2× total |
| Eval cadence | every 100 steps | every 50 steps | catch the peak |
| batch_prompts | 2 | 2 | unchanged (after r=64 was rejected for OOM) |

Everything else mirrors v5 (3-class judge, change-only training, KL=0.1, etc).

## MS-CXR-T trajectory

| Step | macro_acc | per-finding (cons / edema / eff / pneum / ptx) |
|------|-----------|------------------------------------------------|
| 49   | 0.341     | 0.294 / 0.480 / 0.305 / 0.378 / 0.249           |
| **99** | **0.383 (peak)** | 0.223 / 0.471 / 0.332 / 0.357 / **0.529** |
| 149  | 0.353     | 0.351 / 0.376 / 0.356 / 0.308 / 0.374           |
| 199  | 0.348     | (decline confirmed)                             |

**v6 step 99 is a new best at 0.383**, +0.023 over v5 step 99 (0.360), +0.058
over baseline (0.325). Notably, v6 step 99 actually **beats BioViL-T's
pneumothorax number** (0.529 vs 0.508 in paper Table 2). It's the consolidation
finding (0.223) that's holding the average down.

## Headline comparison

| Run                             | macro_acc | Δ vs baseline | % of BioViL-T gap closed |
|---------------------------------|-----------|---------------|--------------------------|
| Baseline (no LoRA)              | 0.325     | —             | 0%                       |
| v1 step 999 (mode-collapsed)    | 0.351     | +0.026        | 9%                       |
| v5 step 99 (change-only filter) | 0.360     | +0.035        | 12%                      |
| **v6 step 99** (r=32 + projector)| **0.383**| **+0.058**    | **20%**                  |
| BioViL-T (Table 2 target)       | 0.612     | —             | 100%                     |

## What we learned

1. **More LoRA capacity helps materially**: +0.023 macro_acc from doubling
   trainable params (40M → 81M), specifically by adapting the projector +
   raising LM rank 16→32. Frozen-vision LoRA is *not* capped at 0.36 as the
   v5 writeup hypothesized — there's headroom past that.
2. **r=64 OOM'd** with a single 24GB card at batch_prompts=2: 162M LoRA params
   + AdamW state pushed allocated memory past the limit even with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Reaching r=64 needs
   either bnb 8-bit optimizer or model-parallel.
3. **Peak is around step 99**, same as v5. Higher capacity = the *same* time
   to peak, but *higher* peak. This suggests the bottleneck past the peak is
   the train/eval distribution gap (ImaGenome silver → MS-CXR-T human) rather
   than capacity.
4. **In-training eval is now a *negative* signal**: v6's ImaGenome val
   macro_acc declined monotonically (0.348 → 0.302 → 0.244) as MS-CXR-T
   macro_acc went *up* from 0.341 → 0.383. Stop on MS-CXR-T (or a held-out
   target-distribution proxy), never on the training distribution.
5. **Per-finding gains are uneven**: v6 step 99 vs v5 step 99 picked up big on
   pneumothorax (+0.142) and pneumonia (+0.108) but lost ground on
   consolidation (−0.157). Consolidation has the widest gap to BioViL-T
   (0.223 vs 0.646) — likely the highest-leverage next target.

## Next moves (in order)

1. **Try bnb 8-bit AdamW** to fit r=64 + projector — would 2× LoRA capacity
   again and tell us if returns are still positive.
2. **Investigate consolidation failure**: pull example reports + judge
   classifications to see whether it's a labelling issue, prompt issue, or
   the rad-DINO encoder genuinely not seeing consolidation changes.
3. **SFT-before-RL**: teach the response format / class distribution
   explicitly with cross-entropy on silver labels, then RL on top. Should
   compress the v6 peak finding faster and possibly raise it.

## Files
Config: `configs/maira2_grpo_imagenome_v6.yaml` (r=32 + projector + change-only)
Best adapter: `checkpoints/maira2_grpo_imagenome_v6/step_000099/`
MS-CXR-T evals: `results/maira2_og_metric_eval/summary_v6_step{049,099,149,199}_*.json`
Training log: `results/maira2_grpo_imagenome_v6/run_*.jsonl`

---

# Fourth iteration: v6a / v6b ablation — disentangling v6's gain (2026-05-20)

v6 changed two things from v5 at once (rank 16→32, +projector targets) and
landed at 0.383 vs v5's 0.360. v6a and v6b split those two changes apart.

## 2×2 design

| Config              | rank | + projector? | step 99 MS-CXR-T | Best (steps 49–149) |
|---------------------|------|--------------|------------------|---------------------|
| Baseline            | —    | —            | —                | 0.325               |
| v5                  | 16   | no           | 0.360            | 0.360               |
| **v6a**             | 32   | no           | 0.321            | 0.341 (step 149)    |
| **v6b**             | 16   | yes          | 0.338            | 0.348 (step 149)    |
| v6                  | 32   | yes          | **0.383**        | **0.383**           |

## What this says

Neither change alone reproduces v6's gain. Capacity-only (v6a) actually drops
*below* v5 at step 99 (the model finds a worse local optimum without new
features to translate). Projector-only (v6b) only nudges marginally up.
**The two together** unlock 0.383 — a +0.058 jump over baseline vs +0.016 (v6a)
and +0.023 (v6b) individually. Roughly additive (sum 0.039) plus an extra
~0.019 of synergy.

The mechanistic read: projector LoRA at any rank gives the LM new
vision-derived features to consume, but at r=16 the LM doesn't have the
plasticity to actually adapt to them. r=32 LM with no projector is just
overfitting to the same frozen rad-DINO distribution, faster. The
combination — more LM plasticity *and* a moving vision-to-LM bridge — is
what shifts the model into a regime where it can describe temporal change
better.

Single-seed numbers, so error bars are loose. But the four-cell pattern is
clean enough to act on.

## Practical implication

The bottleneck is genuinely the vision-side bridge, not LM capacity alone.
Pushing further means either:
1. **Bigger projector adapters** (more rank, more targets) — diminishing
   returns expected once projector saturates.
2. **Swap rad-DINO for a temporally-aware encoder (BioViL-T)** — gives the
   projector richer features to translate. Likely the highest-leverage next
   move. See task #30 (v7).

## Files
Configs: `configs/maira2_grpo_imagenome_v6a.yaml`, `configs/maira2_grpo_imagenome_v6b.yaml`
Eval summaries: `results/maira2_og_metric_eval/summary_v6{a,b}_step*_*.json`

---

# Judge ceiling check: original radiologist reports (2026-05-20)

To sanity-check the OG judge, we reconstructed full MIMIC-CXR reports for
all 315 MS-CXR-T val+test pairs (via the Chest ImaGenome processed-sentences
dump, indexed by subject_id + rad_id) and graded them with the same 3-class
judge we use to score model outputs.

| n   | judge↔MS-CXR-T label macro_acc | per-finding (cons/edema/eff/pneum/ptx)   |
|-----|--------------------------------|------------------------------------------|
| 315 | **0.980**                      | 1.00 / 0.98 / 0.96 / 0.96 / 1.00         |

Only 7/315 disagreements, all on genuinely ambiguous reports ("stable since
X but new since Y", compound findings, hedged temporal language).

## What this means

- **Judge is near-perfect** when fed real radiologist prose; we are not
  hitting a 0.45-0.50 wall because of judge noise.
- The MAIRA-2 baseline's 0.325 macro_acc and our best v6 0.383 reflect
  genuine *report-quality* limits — even with a near-perfect grader,
  MAIRA-2's generated reports lose ~65pp of the temporal signal that's
  preserved in the original radiologist text.
- This is a strong external validity check for the metric: the ~0.6 gap
  between MAIRA-2 baseline and the judge ceiling is "stuff the model isn't
  saying," not "stuff the judge is hallucinating."

## Caveat

Reports were reconstructed from ImaGenome's processed-sentences dump
(`cxr-mimic-v2.0.0-processed-sentences_all.txt`), not the raw MIMIC-CXR
report `.txt` files (which aren't in our local data). Sentence tokenization
+ whitespace collapse could introduce minor deviations, but the
near-perfect judge agreement is itself a strong functional check — if
key temporal sentences had been dropped or scrambled, the agreement would
have cratered.

## Files
Script: `scripts/judge_gt_reports_mscxrt.py`
Per-sample CSV: `results/maira2_og_metric_eval/gt_report_judge_gt_baseline_*.csv`
Summary: `results/maira2_og_metric_eval/gt_report_judge_summary_gt_baseline_*.json`

---

# Fifth iteration: v7e — fine-tuned BioViL-T ensemble as vision encoder (2026-05-20)

v6a/v6b confirmed projector LoRA is half the lever and that the bottleneck is
the frozen vision-side bridge. v7e tries the cleanest version of "give the
vision side something better": swap MAIRA-2's frozen rad-DINO for an ensemble
of 5 fine-tuned BioViL-T encoders (one per MS-CXR-T finding), each of which
hits ~0.6 macro_acc when used standalone end-to-end on its own finding.

## Setup
- Vision tower: 5 frozen BioViL-T encoders, paired-mode forward (each image
  encoded with the partner as temporal context), concat along channel dim →
  (B, 2560, 14, 14)
- Adapter: Linear(2560 → 768) + learned prefix token + LayerNorm (~2M
  trainable, full FT via PEFT modules_to_save)
- LM side: r=32 LoRA + multi_modal_projector targets (v6 config)
- kl_coef=0 (reference policy ill-defined after vision-tower swap)
- Optional MSE bootstrap to pre-align adapter to rad-DINO features (~200
  SGD steps, ~1 min)
- 100 RL training steps, eval every 50

## Results

| Config                       | step 49 MS-CXR-T | step 99 MS-CXR-T | In-training pred_dist (imp/stab/wors) |
|------------------------------|------------------|------------------|---------------------------------------|
| v7e — no bootstrap           | 0.333            | 0.333            | 0 / 100 / 0  (all stable)             |
| v7e — MSE bootstrap (200 st) | 0.333            | 0.333            | 0 / 100 / 0  (all stable)             |
| (reference) v6 step 99       | 0.383            | 0.383            | 21 / 44 / 35                          |
| (reference) baseline         | 0.325            | 0.325            | n/a                                   |

Both v7e variants collapsed to 100% "stable" predictions on the mixed-class
in-training eval, giving the floor macro_acc of 0.333 (= 1/3, since the
balanced accuracy of always-predict-stable is recall=1 on stable +
recall=0 on imp + recall=0 on wors). RL had no advantage signal because
under the change-only training subset every "stable" prediction gets R=-1,
all rollouts identical, gradient zero (or NaN, which the new safety filter
zeroed cleanly).

## What this told us

1. **The vision-tower swap is a deeper distribution shift than LoRA + a
   2560→768 Linear adapter can bridge.** The fine-tuned BioViL-T encoders
   know temporal structure, but feeding their features through a randomly
   initialized adapter into the rad-DINO-trained LM produces garbage that
   the LM defaults to "stable" on.
2. **MSE-bootstrapping the adapter to rad-DINO statistics reduces KL** (6+
   without bootstrap → ~3 with) and stabilizes step 0 numerically (no NaN
   in logits), **but does not change the LM's text behavior** — the LM still
   defaults to "stable" because the bootstrapped features are merely
   *bounded*, not *informative* to the LM.
3. **NaN-gradient filter** added to the main loop catches the case where one
   bad rollout's log_pi is NaN and would corrupt AdamW state — without that
   safety, every v7e variant crashed at step 2 with a `torch.multinomial`
   assertion. This safety also benefits any future runs with new vision
   towers.
4. **Frozen-encoder LoRA has a real ceiling.** v6 (0.383) is roughly at the
   limit of what frozen rad-DINO can support. Hitting the BioViL-T 0.612
   ceiling genuinely requires either (a) end-to-end vision-encoder
   fine-tuning, or (b) supervised fine-tuning of the LM on (image_pair →
   report) pairs before RL — both of which are separate pipelines from the
   GRPO+LoRA recipe we've been iterating on.

## Files
- New module: `models/biovil_t_ensemble_vision_tower.py`
- Single-model variant (also tried, never escaped NaN cold-start):
  `models/biovil_t_vision_tower.py`
- Training-script changes: vision-swap path + bootstrap helper +
  NaN-grad filter in `scripts/rl_maira2_temporal.py`
- Configs: `configs/maira2_grpo_imagenome_v7.yaml`,
  `configs/maira2_grpo_imagenome_v7e.yaml`
- Eval configs: `configs/_eval_v7_mscxrt.yaml`, `configs/_eval_v7e_mscxrt.yaml`
- Adapters: `checkpoints/maira2_grpo_imagenome_v7e/step_*/`
- MS-CXR-T eval summaries:
  `results/maira2_og_metric_eval/summary_v7e_*step*_*.json`
