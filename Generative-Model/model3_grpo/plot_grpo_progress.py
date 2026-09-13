#!/usr/bin/env python3
"""
plot_grpo_progress.py
Parses grpo_run.log (the authoritative per-step reward/loss trace) and
compounds_generated.csv (novel-discovery trend) after a GRPO run finishes,
produces 600 DPI plots + CSV summaries for both.

Usage:
    python3 plot_grpo_progress.py --log grpo_run.log \
        --compounds grpo_checkpoints/compounds_generated.csv --outdir grpo_plots
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STEP_LINE_RE = re.compile(
    r"step (\d+)/(\d+) mean_reward=([\d.\-]+) std=([\d.\-]+) "
    r"pg_loss=([\d.\-]+) kl=([\d.\-]+) total_loss=([\d.\-]+) time=([\d.\-]+)s"
)
TERMS_LINE_RE = re.compile(
    r"reward terms \(mean\): model2=([\d.\-]+) charge=([\d.\-]+) length=([\d.\-]+) "
    r"hmom=([\d.\-]+) div=([\d.\-]+)"
)


def parse_grpo_log(path):
    """Returns a DataFrame, one row per logged step, joining the step-summary
    line with its following reward-terms line."""
    rows = []
    pending = None
    with open(path) as f:
        for line in f:
            m = STEP_LINE_RE.search(line)
            if m:
                pending = {
                    "step": int(m.group(1)),
                    "total_steps": int(m.group(2)),
                    "mean_reward": float(m.group(3)),
                    "std_reward": float(m.group(4)),
                    "pg_loss": float(m.group(5)),
                    "kl": float(m.group(6)),
                    "total_loss": float(m.group(7)),
                    "time_sec": float(m.group(8)),
                }
                continue
            m = TERMS_LINE_RE.search(line)
            if m and pending is not None:
                pending.update({
                    "model2": float(m.group(1)),
                    "charge": float(m.group(2)),
                    "length": float(m.group(3)),
                    "hmom": float(m.group(4)),
                    "diversity_penalty": float(m.group(5)),
                })
                rows.append(pending)
                pending = None
    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return df


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_reward_trend(df, outpath, dpi):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["step"], df["mean_reward"], linewidth=1.6, color="#2166AC", label="Mean reward")
    ax.fill_between(df["step"],
                     df["mean_reward"] - df["std_reward"],
                     df["mean_reward"] + df["std_reward"],
                     alpha=0.15, color="#2166AC", label="±1 std (within group)")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("Reward")
    ax.set_title("GRPO training: mean group reward over time")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_reward_terms(df, outpath, dpi):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, color, label in [
        ("model2", "#B2182B", "Model2 score"),
        ("charge", "#2166AC", "Charge score"),
        ("length", "#1B7837", "Length score"),
        ("hmom", "#762A83", "Hydrophobic moment"),
        ("diversity_penalty", "#D6604D", "Diversity penalty"),
    ]:
        ax.plot(df["step"], df[col], linewidth=1.3, label=label, color=color)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("Term value")
    ax.set_title("GRPO training: individual reward-term trends")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_kl_and_loss(df, outpath, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(df["step"], df["kl"], linewidth=1.4, color="#D6604D")
    axes[0].set_xlabel("GRPO step")
    axes[0].set_ylabel("KL divergence (vs. frozen reference)")
    axes[0].set_title("Policy drift from Stage 2 reference")
    style_axis(axes[0])

    axes[1].plot(df["step"], df["pg_loss"], linewidth=1.2, color="#2166AC", label="PG loss")
    axes[1].plot(df["step"], df["total_loss"], linewidth=1.2, color="#B2182B", label="Total loss")
    axes[1].set_xlabel("GRPO step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training loss components")
    axes[1].legend(frameon=False)
    style_axis(axes[1])

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_novel_discovery(compounds_path, outpath, dpi):
    df = pd.read_csv(compounds_path)
    counts_by_step = df.groupby("step").size().reset_index(name="new_novel_count")
    counts_by_step["cumulative_novel"] = counts_by_step["new_novel_count"].cumsum()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(counts_by_step["step"], counts_by_step["cumulative_novel"],
                 linewidth=1.6, color="#1B7837")
    axes[0].set_xlabel("GRPO step")
    axes[0].set_ylabel("Cumulative novel sequences")
    axes[0].set_title("Novel-compound accumulation")
    style_axis(axes[0])

    axes[1].plot(counts_by_step["step"], counts_by_step["new_novel_count"],
                 linewidth=1.0, color="#2166AC", alpha=0.7)
    axes[1].set_xlabel("GRPO step")
    axes[1].set_ylabel("New novel sequences this step")
    axes[1].set_title("Per-step novelty rate (watch for decline = mode collapse)")
    style_axis(axes[1])

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")
    return counts_by_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="grpo_run.log")
    ap.add_argument("--compounds", default="grpo_checkpoints/compounds_generated.csv")
    ap.add_argument("--outdir", default="grpo_plots")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print(f"Parsing {args.log}...")
    df = parse_grpo_log(args.log)
    print(f"Parsed {len(df)} logged steps")
    if df.empty:
        raise RuntimeError("Zero rows parsed from the log - check --log_every was set to "
                            "something that actually produced matching lines.")
    df.to_csv(outdir / "grpo_step_metrics.csv", index=False)

    plot_reward_trend(df, outdir / "grpo_reward_trend.png", args.dpi)
    plot_reward_terms(df, outdir / "grpo_reward_terms.png", args.dpi)
    plot_kl_and_loss(df, outdir / "grpo_kl_and_loss.png", args.dpi)

    if Path(args.compounds).exists():
        counts_by_step = plot_novel_discovery(args.compounds, outdir / "grpo_novel_discovery.png", args.dpi)
        counts_by_step.to_csv(outdir / "grpo_novel_discovery.csv", index=False)
    else:
        print(f"[warn] {args.compounds} not found - skipping novel-discovery plot")

    print(f"\nAll outputs saved to {outdir}/")
    print(f"\nFinal step summary:\n{df.iloc[-1]}")


if __name__ == "__main__":
    main()
