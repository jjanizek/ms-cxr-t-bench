"""Main training entrypoint for Protocol A (linear probe).

Usage:
    python scripts/train.py --config configs/resnet_protocol_a.yaml
    python scripts/train.py --config configs/biovil_t_protocol_a.yaml --seed 42
"""
import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from data.dataset import load_labels, PairDataset, FINDINGS, LABEL_MAP


def load_model(cfg: dict):
    """Instantiate model from config."""
    model_name = cfg["model"]["name"]
    model_kwargs = cfg["model"].get("kwargs", {})
    device = cfg.get("device", "cpu")

    if model_name == "resnet_baseline":
        from models.resnet_baseline import ResNetBaseline
        model = ResNetBaseline(**model_kwargs)
    elif model_name == "densenet_baseline":
        from models.densenet_baseline import DenseNetBaseline
        model = DenseNetBaseline(**model_kwargs)
    elif model_name == "biovil":
        from models.biovil import BioViL
        model = BioViL(device=device, **model_kwargs)
    elif model_name == "biovil_t":
        from models.biovil_t import BioViLT
        model = BioViLT(device=device, **model_kwargs)
    elif model_name == "medst":
        from models.medst import MedST
        model = MedST(device=device, **model_kwargs)
    elif model_name == "google_cxr":
        from models.google_cxr import GoogleCXR
        model = GoogleCXR(device=device, **model_kwargs)
    elif model_name == "ours":
        from models.ours import OurModel
        model = OurModel(device=device, **model_kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def extract_features_cached(
    model,
    df,
    cache_path: Path,
    images_root: str,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract or load cached concatenated pair features for all rows in df."""
    if cache_path.exists():
        logger.info("Loading cached features from %s", cache_path)
        return np.load(cache_path)

    logger.info("Extracting features for %d pairs...", len(df))
    from torch.utils.data import DataLoader
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = PairDataset(df, images_root, transform=tf)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    all_feats = []
    model.eval()
    with torch.no_grad():
        for img1, img2, _ in loader:
            feats = model.encode_image_pair(img1.to(device), img2.to(device))
            all_feats.append(feats.cpu().numpy())

    features = np.concatenate(all_feats, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    logger.info("Saved features to %s", cache_path)
    return features


def run_protocol_a(cfg: dict, model, df, seeds: list[int]) -> list[dict]:
    """Run Protocol A for all findings and seeds."""
    from evaluation.temporal_cls import run_protocol_a as _run

    model_name = cfg["model"]["name"]
    features_dir = Path("data/features") / model_name
    images_root = cfg["data"]["images_root"]
    device = cfg.get("device", "cpu")
    preextracted = cfg.get("features_preextracted", False)

    all_results = []
    for finding in FINDINGS:
        df_finding = df[df["finding"] == finding].copy()
        if len(df_finding) == 0:
            logger.warning("No data for finding: %s", finding)
            continue

        cache = features_dir / f"{finding}_all.npy"

        if preextracted:
            # Features were produced offline (e.g. by extract_google_cxr_features.py)
            if not cache.exists():
                raise FileNotFoundError(
                    f"Pre-extracted feature cache not found: {cache}\n"
                    "Run: python scripts/extract_google_cxr_features.py"
                )
            logger.info("Loading pre-extracted features from %s", cache)
            all_feats = np.load(cache)
        else:
            all_feats = extract_features_cached(model, df_finding, cache, images_root, device)
        all_labels = df_finding["label"].values
        subjects = df_finding["subject_id"].values

        for seed in seeds:
            split_path = Path("data/splits") / f"split_seed{seed}.json"
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Split file not found: {split_path}. "
                    "Run: python scripts/make_splits.py first."
                )
            with open(split_path) as f:
                split = json.load(f)

            def select(split_subjects):
                mask = np.isin(subjects, split_subjects)
                return all_feats[mask], all_labels[mask]

            features = {
                "train": select(split["train"]),
                "val": select(split["val"]),
                "test": select(split["test"]),
            }

            # Skip if any split has no data for this finding
            for part, (X, y) in features.items():
                if len(X) == 0:
                    logger.warning(
                        "Empty %s split for %s seed=%d", part, finding, seed
                    )

            result = _run(
                features, finding, seed,
                device=device,
                probe_type=cfg.get("probe_type", "linear"),
                **cfg.get("probe_kwargs", {}),
            )
            all_results.append(result)
            logger.info(
                "Seed %d | %s | macro_acc=%.3f macro_f1=%.3f",
                seed, finding, result["macro_acc"], result["macro_f1"],
            )

    return all_results


def summarise(results: list[dict]) -> dict:
    """Average Protocol A results across seeds, per finding and overall."""
    import pandas as pd

    df = pd.DataFrame(results)
    summary = {}
    for finding in FINDINGS:
        sub = df[df["finding"] == finding]
        if len(sub) == 0:
            continue
        summary[finding] = {
            k: {"mean": float(sub[k].mean()), "std": float(sub[k].std())}
            for k in ["macro_acc", "macro_f1"]
            if k in sub.columns
        }

    # Average across findings (per seed, then aggregate)
    for k in ["macro_acc", "macro_f1"]:
        if k in df.columns:
            # Mean per seed across findings, then mean±std over seeds
            per_seed = df.groupby("seed")[k].mean()
            summary[f"avg_{k}"] = {
                "mean": float(per_seed.mean()),
                "std": float(per_seed.std()),
            }
    return summary


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
    parser.add_argument("--seed", type=int, default=None, help="Override seeds from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [42, 123, 456, 789])
    protocol = cfg.get("protocol", "A")

    logger.info("Config: %s | Protocol: %s | Seeds: %s", args.config, protocol, seeds)

    preextracted = cfg.get("features_preextracted", False)
    model = None if preextracted else load_model(cfg)
    df = load_labels(cfg["data"]["labels"], images_root=cfg["data"]["images_root"])
    logger.info(
        "Loaded %d (pair, finding) rows across %d subjects",
        len(df), df["subject_id"].nunique(),
    )

    if protocol == "A":
        results = run_protocol_a(cfg, model, df, seeds)
    else:
        raise NotImplementedError(
            f"Protocol {protocol} not implemented in train.py. "
            "Use evaluate.py for Protocol B/C."
        )

    summary = summarise(results)
    def _fmt(v):
        return f"{v['mean']:.3f}±{v['std']:.3f}"

    summary_log = {}
    for k, vv in summary.items():
        if "mean" in vv:  # avg_macro_acc / avg_macro_f1 — flat {mean, std}
            summary_log[k] = _fmt(vv)
        else:             # per-finding — {metric: {mean, std}}
            summary_log[k] = {m: _fmt(v) for m, v in vv.items()}
    logger.info("Summary:\n%s", json.dumps(summary_log, indent=2))

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"{cfg['model']['name']}_protocol{protocol}_{timestamp}.json"
    payload = {
        "config": cfg,
        "config_path": args.config,
        "git_hash": git_hash(),
        "timestamp": timestamp,
        "seeds": seeds,
        "results": results,
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
