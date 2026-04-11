"""Run all model × protocol combinations in sequence.

Usage:
    python scripts/run_sweep.py --protocol A
    python scripts/run_sweep.py --protocol A --models resnet_baseline biovil_t
    python scripts/run_sweep.py --dry-run
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROTOCOL_A_CONFIGS = [
    "configs/resnet_protocol_a.yaml",
    "configs/densenet_protocol_a.yaml",
    "configs/biovil_protocol_a.yaml",
    "configs/biovil_t_protocol_a.yaml",
    "configs/medst_protocol_a.yaml",
    "configs/google_cxr_protocol_a.yaml",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="A", choices=["A", "B", "sentence_sim"])
    parser.add_argument("--models", nargs="*", help="Filter to specific model names")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.protocol == "A":
        configs = PROTOCOL_A_CONFIGS
    else:
        raise NotImplementedError(f"Sweep for protocol {args.protocol} not yet configured.")

    if args.models:
        configs = [c for c in configs if any(m in c for m in args.models)]

    missing = [c for c in configs if not Path(c).exists()]
    if missing:
        print(f"WARNING: Missing config files: {missing}")

    for cfg_path in configs:
        if not Path(cfg_path).exists():
            print(f"SKIP (missing): {cfg_path}")
            continue
        cmd = [sys.executable, "scripts/train.py", "--config", cfg_path]
        print(f"{'DRY-RUN' if args.dry_run else 'RUNNING'}: {' '.join(cmd)}")
        if not args.dry_run:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"FAILED: {cfg_path} (exit {result.returncode})")


if __name__ == "__main__":
    main()
