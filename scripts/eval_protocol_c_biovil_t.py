"""Protocol C evaluation for BioViL-T: zero-shot temporal image classification.

Replicates the evaluation from Bannur et al., CVPR 2023.

Each (prior, current) image pair is encoded by BioViL-T's MultiImageEncoder
into a 128-dim l2-normalised embedding.  Text templates for each progression
class are encoded by BioViL-T's CXR-BERT and averaged into a single embedding.
The predicted class is the one with highest cosine similarity.

Metrics reported: standard accuracy, macro-accuracy, macro-F1 (per finding and
averaged), to allow comparison with the paper regardless of which metric they use.

Usage:
    python scripts/eval_protocol_c_biovil_t.py
    python scripts/eval_protocol_c_biovil_t.py --device cuda
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text templates per progression class.
# Multiple prompts per class are averaged before l2-normalisation, following
# the standard CLIP zero-shot protocol (Radford et al., 2021).
# Templates use clinical radiology language; finding name is substituted in.
# ---------------------------------------------------------------------------

TEMPLATES = {
    "improving": [
        "Interval improvement in {finding}.",
        "There has been improvement in {finding} since the prior study.",
        "The {finding} has improved.",
        "Decrease in {finding} compared to the prior examination.",
        "The {finding} appears improved.",
    ],
    "stable": [
        "No significant interval change in {finding}.",
        "The {finding} is stable compared to the prior study.",
        "The {finding} is unchanged.",
        "Unchanged {finding} since the prior examination.",
        "The {finding} remains similar to before.",
    ],
    "worsening": [
        "Interval worsening of {finding}.",
        "There has been worsening of {finding} since the prior study.",
        "The {finding} has worsened.",
        "Increase in {finding} compared to the prior examination.",
        "The {finding} appears worse.",
    ],
}

# Clinical names used in text prompts (maps our finding column values)
FINDING_NAMES = {
    "consolidation":     "consolidation",
    "edema":             "pulmonary edema",
    "pleural_effusion":  "pleural effusion",
    "pneumonia":         "pneumonia",
    "pneumothorax":      "pneumothorax",
}

CLASS_ORDER = ["improving", "stable", "worsening"]  # matches LABEL_MAP 0,1,2


def load_engines(device: str):
    from health_multimodal.image.utils import get_image_inference, ImageModelType
    from health_multimodal.image.model.pretrained import BIOMED_VLP_BIOVIL_T, BIOVIL_T_COMMIT_TAG
    from health_multimodal.text.model import CXRBertModel
    from transformers import AutoTokenizer

    logger.info("Loading BioViL-T image engine...")
    image_engine = get_image_inference(ImageModelType.BIOVIL_T)
    image_engine.model = image_engine.model.to(device).eval()

    logger.info("Loading BioViL-T text model (via AutoTokenizer)...")
    # CXRBertTokenizer.encode only handles single strings; AutoTokenizer with
    # trust_remote_code=True gives a full tokenizer with batch_encode_plus.
    tokenizer = AutoTokenizer.from_pretrained(
        BIOMED_VLP_BIOVIL_T, revision=BIOVIL_T_COMMIT_TAG, trust_remote_code=True
    )
    text_model = CXRBertModel.from_pretrained(BIOMED_VLP_BIOVIL_T, revision=BIOVIL_T_COMMIT_TAG)
    text_model = text_model.to(device).eval()

    return image_engine, (tokenizer, text_model)


@torch.no_grad()
def get_class_text_embeddings(text_engine, finding: str, device: str) -> torch.Tensor:
    """Return (3, 128) tensor: one l2-normalised embedding per class.

    Encodes all prompts for a class in one batch, averages unnormalised
    embeddings, then l2-normalises — the standard CLIP zero-shot protocol.
    """
    tokenizer, text_model = text_engine
    finding_name = FINDING_NAMES[finding]
    class_embs = []
    for cls in CLASS_ORDER:
        prompts = [t.format(finding=finding_name) for t in TEMPLATES[cls]]
        # strip trailing punctuation (hi-ml convention)
        prompts = [p.rstrip("!?.") for p in prompts]
        enc = tokenizer(
            prompts, add_special_tokens=True, padding="longest",
            return_tensors="pt", truncation=True,
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        embs = text_model.get_projected_text_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            normalize_embeddings=False,
        )  # (N, 128)
        avg = embs.mean(dim=0)
        avg = F.normalize(avg, dim=0)
        class_embs.append(avg)
    return torch.stack(class_embs)  # (3, 128)


@torch.no_grad()
def encode_pair(image_engine, path_prior: Path, path_curr: Path, device: str) -> torch.Tensor:
    """Encode (prior, current) pair → (128,) l2-normalised image embedding."""
    model = image_engine.model
    transform = image_engine.transform

    img1, _ = image_engine.load_and_transform_input_image(path_prior, transform)
    img2, _ = image_engine.load_and_transform_input_image(path_curr, transform)
    img1, img2 = img1.to(device), img2.to(device)

    patch_fused, avg_pooled = model.encoder(
        current_image=img2,
        previous_image=img1,
        return_patch_embeddings=True,
    )
    out = model.forward_post_encoder(patch_fused, avg_pooled)
    emb = F.normalize(out.projected_global_embedding, dim=-1)  # (1, 128)
    return emb[0]  # (128,)


def evaluate_finding(image_engine, text_engine, df_f, images_root, device, finding):
    """Run zero-shot classification for one finding. Returns metrics dict."""
    class_embs = get_class_text_embeddings(text_engine, finding, device)  # (3, 128)

    preds, labels = [], []
    for _, row in tqdm(df_f.iterrows(), total=len(df_f), desc=finding):
        from data.dataset import dicom_id_to_filename
        path_prior = images_root / dicom_id_to_filename(row["previous_dicom_id"])
        path_curr = images_root / dicom_id_to_filename(row["dicom_id"])
        img_emb = encode_pair(image_engine, path_prior, path_curr, device)  # (128,)
        sims = class_embs @ img_emb  # (3,)
        pred = sims.argmax().item()
        preds.append(pred)
        labels.append(int(row["label"]))

    y_pred = np.array(preds)
    y_true = np.array(labels)

    acc = float(accuracy_score(y_true, y_pred))
    macro_acc = float(balanced_accuracy_score(y_true, y_pred))  # = macro recall
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class breakdown
    per_class = {}
    for i, cls in enumerate(CLASS_ORDER):
        mask = y_true == i
        if mask.sum() > 0:
            per_class[cls] = float((y_pred[mask] == i).mean())

    return {
        "finding": finding,
        "n": len(y_true),
        "acc": acc,
        "macro_acc": macro_acc,
        "macro_f1": macro_f1,
        "per_class_recall": per_class,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/raw/ms_cxr_t_labels.csv")
    parser.add_argument("--images_root", default="data/raw/images")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--finding", default=None, help="Evaluate only this finding")
    parser.add_argument("--out", default="results/biovil_t_protocol_c.json")
    args = parser.parse_args()

    from data.dataset import load_labels, FINDINGS

    df = load_labels(args.labels, images_root=args.images_root)
    images_root = Path(args.images_root)
    logger.info("Loaded %d (pair, finding) rows across %d subjects", len(df), df["subject_id"].nunique())

    image_engine, text_engine = load_engines(args.device)  # text_engine = (tokenizer, text_model)

    findings = [args.finding] if args.finding else FINDINGS
    all_results = []

    for finding in findings:
        df_f = df[df["finding"] == finding].reset_index(drop=True)
        logger.info("Finding: %s (%d pairs)", finding, len(df_f))
        result = evaluate_finding(image_engine, text_engine, df_f, images_root, args.device, finding)
        all_results.append(result)
        logger.info(
            "  acc=%.3f  macro_acc=%.3f  macro_f1=%.3f  per_class=%s",
            result["acc"], result["macro_acc"], result["macro_f1"],
            {k: f"{v:.2f}" for k, v in result["per_class_recall"].items()},
        )

    # Summary
    avg_acc = np.mean([r["acc"] for r in all_results])
    avg_macro_acc = np.mean([r["macro_acc"] for r in all_results])
    avg_macro_f1 = np.mean([r["macro_f1"] for r in all_results])
    logger.info(
        "Average over %d findings — acc=%.3f  macro_acc=%.3f  macro_f1=%.3f",
        len(all_results), avg_acc, avg_macro_acc, avg_macro_f1,
    )

    payload = {
        "model": "biovil_t",
        "protocol": "C",
        "templates": TEMPLATES,
        "finding_names": FINDING_NAMES,
        "results": all_results,
        "summary": {
            "avg_acc": float(avg_acc),
            "avg_macro_acc": float(avg_macro_acc),
            "avg_macro_f1": float(avg_macro_f1),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Results saved to %s", args.out)


if __name__ == "__main__":
    main()
