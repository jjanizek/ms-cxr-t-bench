# MS-CXR-T Model Benchmarking

## Project Goal
Benchmark multiple vision-language and vision-only models on MS-CXR-T temporal chest X-ray tasks, with Protocol A (70/10/20 fixed split) as the primary head-to-head comparison. Additional protocols (cross-validation, zero-shot) are secondary and used for comparability with published results. At project end, we evaluate our own foundation model. We are also generating a supplementary label set (~1-2K samples) covering critical findings and other annotations not in the original benchmark.

## Environment
- GPU server: 2x RTX 4090 (accessed via `ssh -p 1017 joseph@10.0.0.105`)
- Python: use conda (check `conda env list` for available envs; create one if needed)
- Framework: PyTorch

## Data Paths
- **MS-CXR-T images**: `/data/ms-cxr-t-relabel/`
- **MS-CXR-T temporal classification labels**: `~/Downloads/MS_CXR_T_temporal_image_classification_v1.0.0.csv`
- **MIMIC-CXR metadata**: `~/Downloads/mimic-cxr-2.0.0-metadata.csv`
- **Custom labels**: `data/custom_labels/` (in repo — versioned, added as generated)

## Dataset: MS-CXR-T
- Source: Bannur et al. (CVPR 2023), derived from MIMIC-CXR v2
- Temporal image classification: 1,326 image-pair annotations, 800 subjects
  - 5 findings: Consolidation (201), Edema (266), Pleural Effusion (411), Pneumonia (237), Pneumothorax (211)
  - 3 progression classes: Improving (18%), Stable (40%), Worsening (42%)
- Temporal sentence similarity: 361 sentence pairs (141 paraphrase / 220 contradiction)
  - Subsets: RadGraph (117), Swaps (244)
- NOTE: Some papers report 1,045 pairs after filtering. Document any filtering we apply.

## Primary Evaluation Protocol (Protocol A)
- 70% train / 10% val / 20% test split at the **subject level** (no patient leakage)
- Freeze pretrained encoder, train linear classification head only
- Input: concatenated image-pair features → 3-class prediction (Improving/Stable/Worsening)
- Metrics: **macro-accuracy (%)** per finding and averaged across findings
- Secondary metrics: macro F1, macro AUROC
- Report mean ± std over 4 seeds: [42, 123, 456, 789]
- Used by: HERGen (Wang et al. 2024), ALTA (Lian et al. 2025)

## Secondary Protocols (for published-number comparability)
- **Protocol B (5-fold CV)**: SVM on concatenated features. Used by Med-ST.
- **Protocol C (zero-shot/few-shot)**: Prompt-based or cosine similarity. Used by TempA-VLP, BioViL-T.
- **Sentence similarity**: Cosine similarity + threshold tuning (10-fold CV). Metrics: Accuracy, ROC-AUC on RadGraph and Swaps subsets.
- **Order sensitivity** (stretch goal): Reversed input pairs, consistency scoring (Choi et al., Ko et al.)

## Models to Benchmark
### External / pretrained (priority order)
1. **Google CXR Foundation Model** — primary model of interest
2. **BioViL-T** — canonical MS-CXR-T temporal baseline (Microsoft)
3. **BioViL** — static baseline (Microsoft)
4. **Med-ST** — code: https://github.com/SVT-Yang/MedST
5. **ALTA** — code: https://github.com/DopamineLcy/ALTA
6. **CheXRelNet**
7. ImageNet-pretrained ResNet (frozen + linear head, simple baseline)
8. Random classifier baseline

### Our model
- TBD — stub in `models/ours.py`, will be integrated when ready

## Repo Structure
```
ms-cxr-t-bench/
├── CLAUDE.md
├── configs/              # YAML configs: one per model × protocol
├── data/
│   ├── raw/              # Symlinks to server data paths
│   ├── custom_labels/    # Our supplementary labels (versioned)
│   └── splits/           # Saved train/val/test split CSVs (with seed)
├── models/
│   ├── base.py           # BaseModel ABC: encode_image, encode_image_pair, encode_text
│   ├── google_cxr.py
│   ├── biovil.py
│   ├── biovil_t.py
│   ├── medst.py
│   ├── chexrelnet.py
│   ├── resnet_baseline.py
│   └── ours.py           # Stub for our foundation model
├── evaluation/
│   ├── temporal_cls.py   # Protocol A/B/C temporal image classification
│   ├── sentence_sim.py   # Temporal sentence similarity eval
│   ├── custom_labels.py  # Eval on our custom label set
│   └── metrics.py        # Shared metric computation (macro-acc, F1, AUROC)
├── scripts/
│   ├── train.py          # Main entrypoint: python scripts/train.py --config configs/X.yaml
│   ├── evaluate.py       # Eval-only entrypoint
│   ├── make_splits.py    # Generate and save subject-level splits
│   └── run_sweep.py      # Run all model × protocol combos
├── results/              # JSON: full config + metrics + seed + timestamp per run
├── notebooks/            # Analysis and visualization
├── requirements.txt
└── README.md
```

## Conventions
- Config-driven: `python scripts/train.py --config configs/biovil_t_protocol_a.yaml`
- All models implement `models/base.py` interface:
  - `encode_image(img) -> Tensor`
  - `encode_image_pair(img1, img2) -> Tensor`
  - `encode_text(text) -> Tensor` (where applicable)
- Subject-level splits only — never leak patients across train/val/test
- Results: JSON with full config dump, all metrics, seed, git hash, timestamp
- Clone published repos (Med-ST, ALTA) into `external/` and wrap — don't reimplement
- Watch class imbalance: Improving = 18%. Consider stratified splits.

## Implementation Notes
- MS-CXR-T CSV references MIMIC-CXR study/subject/dicom IDs → need mapping to actual
  image file paths via `mimic-cxr-2.0.0-metadata.csv`
- Images in `/data/ms-cxr-t-relabel/` — verify naming convention and match to metadata
- Google CXR Foundation may need specific preprocessing (resolution, normalization) — check docs
- Default image preprocessing unless model specifies otherwise: resize to 256, center crop to 224, ImageNet normalize
- For linear probing: extract features once per model and cache to `data/features/` to avoid recomputation