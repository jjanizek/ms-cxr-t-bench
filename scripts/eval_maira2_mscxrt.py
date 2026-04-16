"""Run MAIRA-2 on MS-CXR-T image pairs and save generated reports.

Two prompting strategies:
  (A) Standard report generation: prior + current images, default MAIRA-2 prompt.
      Let the model generate a natural findings report.
  (B) Finding-specific temporal prompt: ask about a specific finding's progression
      in the comparison field to encourage focused temporal language.

Each (pair, finding) gets a generated report saved to a JSON file.  The reports
are later processed by llm_judge_temporal.py to extract structured labels.

Usage:
    python scripts/eval_maira2_mscxrt.py
    python scripts/eval_maira2_mscxrt.py --prompt_mode specific --finding edema
    python scripts/eval_maira2_mscxrt.py --device cuda:1
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FINDING_DISPLAY = {
    "consolidation": "consolidation",
    "edema": "pulmonary edema",
    "pleural_effusion": "pleural effusion",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
}


def load_maira(model_name: str, device_map="auto"):
    from transformers import AutoModelForCausalLM, AutoProcessor

    logger.info("Loading MAIRA-2 from %s ...", model_name)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model.eval()
    logger.info("MAIRA-2 loaded. Device map: %s", getattr(model, "hf_device_map", "N/A"))
    return model, processor


@torch.inference_mode()
def generate_temporal_report(
    model, processor, prior_path: Path, curr_path: Path,
    finding: str = None, prompt_mode: str = "standard",
    max_new_tokens: int = 300,
) -> str:
    """Generate a report from a (prior, current) image pair.

    prompt_mode:
      'standard' — natural report gen with prior image, comparison="Prior study available."
      'specific' — targeted: comparison mentions the specific finding to evaluate.
    """
    prior_img = Image.open(prior_path).convert("RGB")
    curr_img = Image.open(curr_path).convert("RGB")

    if prompt_mode == "standard":
        comparison = "Prior study available."
    elif prompt_mode == "specific" and finding:
        display = FINDING_DISPLAY.get(finding, finding)
        comparison = (
            f"Prior study available. "
            f"Compare the current and prior study and describe any change in {display}."
        )
    else:
        comparison = "Prior study available."

    inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=curr_img,
        current_lateral=None,
        prior_frontal=prior_img,
        indication="",
        technique="AP portable chest radiograph.",
        comparison=comparison,
        prior_report=None,
        return_tensors="pt",
        get_grounding=False,
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    prompt_len = inputs["input_ids"].shape[-1]
    decoded = processor.decode(output[0][prompt_len:], skip_special_tokens=True).lstrip()
    parsed = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
    return parsed.strip() if isinstance(parsed, str) else str(parsed).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/maira-2")
    parser.add_argument("--mscxrt_labels", default="data/raw/ms_cxr_t_labels.csv")
    parser.add_argument("--mscxrt_images", default="data/raw/images")
    parser.add_argument("--prompt_mode", choices=["standard", "specific"], default="standard",
                        help="standard: natural report. specific: ask about the finding explicitly.")
    parser.add_argument("--finding", default=None, help="Run only this finding (default: all 5)")
    parser.add_argument("--out_dir", default="data/maira2_reports")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--device", default=None,
                        help="e.g. cuda:0. Default: device_map=auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from data.dataset import load_labels, FINDINGS, dicom_id_to_filename

    findings = [args.finding] if args.finding else FINDINGS
    df = load_labels(args.mscxrt_labels, images_root=args.mscxrt_images)
    logger.info("MS-CXR-T: %d (pair, finding) rows", len(df))

    out_dir = Path(args.out_dir) / args.prompt_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    device_map = args.device if args.device else "auto"
    model, processor = load_maira(args.model_name, device_map=device_map)

    results = []
    images_root = Path(args.mscxrt_images)

    for finding in findings:
        df_f = df[df["finding"] == finding].reset_index(drop=True)
        logger.info("Finding: %s — %d pairs", finding, len(df_f))

        for idx, row in tqdm(df_f.iterrows(), total=len(df_f), desc=finding):
            curr_name = Path(row["dicom_id"]).name
            prior_name = Path(row["previous_dicom_id"]).name
            pair_id = f"{curr_name}_{prior_name}_{finding}"
            out_file = out_dir / f"{pair_id}.json"
            if out_file.exists() and not args.overwrite:
                with open(out_file) as f:
                    results.append(json.load(f))
                continue

            curr_path = images_root / dicom_id_to_filename(row["dicom_id"])
            prior_path = images_root / dicom_id_to_filename(row["previous_dicom_id"])

            if not curr_path.exists() or not prior_path.exists():
                logger.warning("Missing image: %s or %s", curr_path, prior_path)
                continue

            report = generate_temporal_report(
                model, processor, prior_path, curr_path,
                finding=finding, prompt_mode=args.prompt_mode,
                max_new_tokens=args.max_new_tokens,
            )

            record = {
                "dicom_id": row["dicom_id"],
                "previous_dicom_id": row["previous_dicom_id"],
                "finding": finding,
                "ground_truth": row["progression"],
                "label": int(row["label"]),
                "prompt_mode": args.prompt_mode,
                "maira2_report": report,
            }
            with open(out_file, "w") as f:
                json.dump(record, f, indent=2)
            results.append(record)

    # Save consolidated results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    consolidated = out_dir / f"all_reports_{timestamp}.json"
    with open(consolidated, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %d reports to %s", len(results), consolidated)


if __name__ == "__main__":
    main()
