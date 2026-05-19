"""Quick readouts for an RL training run.

Reads results/<run>/run_*.jsonl, prints per-100-step aggregates and saves
a PNG with reward, KL, and grad-norm curves.

    python scripts/plot_rl_curves.py results/maira2_grpo_imagenome
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--window", type=int, default=25, help="Rolling-mean window")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    # Combine all run_*.jsonl files (in case of restarts)
    rows = []
    for jl in sorted(run_dir.glob("run_*.jsonl")):
        rows.extend(json.loads(l) for l in jl.read_text().splitlines() if l.strip())
    if not rows:
        sys.exit(f"No log rows in {run_dir}")
    df = pd.DataFrame(rows)
    n = len(df)
    print(f"Rows: {n}")
    print(f"Cols: {list(df.columns)}")
    print()

    # Per-bucket summary
    bucket = max(1, n // 10)
    print(f"=== Per-{bucket}-step aggregates (so 10 rows max) ===")
    print(f"{'step':>6}  {'R_mean':>7}  {'R_lbl':>7}  {'kl':>7}  "
          f"{'grad_l1':>10}  {'time_s':>7}  {'all_same%':>9}")
    for start in range(0, n, bucket):
        end = min(start + bucket, n)
        sub = df.iloc[start:end]
        all_same_rate = float((sub["reward_std"] < 1e-6).mean())
        print(f"{int(sub['step'].iloc[-1]):>6}  "
              f"{sub['reward_mean'].mean():>7.3f}  "
              f"{sub['r_label_mean'].mean():>7.3f}  "
              f"{sub['kl'].mean():>7.4f}  "
              f"{sub['grad_l1'].mean():>10.0f}  "
              f"{sub['time_s'].mean():>7.1f}  "
              f"{100*all_same_rate:>8.1f}%")
    print()
    print(f"Cumulative: {n} steps, "
          f"all-same rate {100*(df['reward_std']<1e-6).mean():.1f}%, "
          f"avg time/step {df['time_s'].mean():.1f}s")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    w = min(args.window, max(1, n // 5))
    def smooth(s): return s.rolling(window=w, min_periods=1).mean()

    ax = axes[0][0]
    ax.plot(df["step"], df["r_label_mean"], color="C0", alpha=0.25)
    ax.plot(df["step"], smooth(df["r_label_mean"]), color="C0", label=f"R_label (rolling {w})")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("R_label (mean per rollout)")
    ax.legend(loc="lower right")
    ax.set_title("Label reward")

    ax = axes[0][1]
    ax.plot(df["step"], df["reward_std"], color="C2", alpha=0.25)
    ax.plot(df["step"], smooth(df["reward_std"]), color="C2", label=f"reward_std (rolling {w})")
    ax.set_ylabel("Within-group reward std")
    ax.legend(loc="lower right")
    ax.set_title("Advantage signal (std)")

    ax = axes[1][0]
    ax.plot(df["step"], df["kl"], color="C3", alpha=0.4)
    ax.plot(df["step"], smooth(df["kl"]), color="C3", label=f"KL (rolling {w})")
    ax.set_ylabel("KL(policy || ref)")
    ax.set_xlabel("step")
    ax.legend(loc="upper left")
    ax.set_title("Policy drift")

    ax = axes[1][1]
    ax.plot(df["step"], df["grad_l1"], color="C4", alpha=0.4)
    ax.plot(df["step"], smooth(df["grad_l1"]), color="C4", label=f"grad_l1 (rolling {w})")
    ax.set_ylabel("Σ|grad| (LoRA params)")
    ax.set_xlabel("step")
    ax.legend(loc="upper right")
    ax.set_title("Gradient magnitude")

    fig.suptitle(f"{run_dir.name}  —  {n} steps", fontsize=12)
    fig.tight_layout()
    out = run_dir / "curves.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
