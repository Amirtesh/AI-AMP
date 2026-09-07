#!/usr/bin/env python3
"""
Parse Stage 1 / Stage 2 training logs and generate publication-quality
(600 DPI) loss curve plots.

Usage:
    python3 plot_training_curves.py --stage1_log stage1.log --stage2_log stage2.log --outdir plots
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EPOCH_LINE_RE = re.compile(
    r"Epoch (\d+): train_loss=([\d.]+) val_loss=([\d.]+) time=([\d.]+)min"
)
STEP_LINE_RE = re.compile(
    r"epoch (\d+) step (\d+)/(\d+) loss=([\d.]+) lr=([\d.eE+-]+)"
)


def parse_log(path):
    """Returns (epoch_df, step_df) from a training log file."""
    epoch_rows, step_rows = [], []
    with open(path) as f:
        for line in f:
            m = EPOCH_LINE_RE.search(line)
            if m:
                epoch_rows.append({
                    "epoch": int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "val_loss": float(m.group(3)),
                    "time_min": float(m.group(4)),
                })
                continue
            m = STEP_LINE_RE.search(line)
            if m:
                epoch, step, total_steps, loss, lr = m.groups()
                step_rows.append({
                    "epoch": int(epoch),
                    "step": int(step),
                    "total_steps": int(total_steps),
                    "loss": float(loss),
                    "lr": float(lr),
                    "global_step": int(epoch) * int(total_steps) + int(step),
                })
    epoch_df = pd.DataFrame(epoch_rows).sort_values("epoch").reset_index(drop=True)
    step_df = pd.DataFrame(step_rows).sort_values("global_step").reset_index(drop=True)
    return epoch_df, step_df


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_epoch_loss(epoch_df, title, outpath, dpi):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(epoch_df["epoch"], epoch_df["train_loss"], marker="o", markersize=4,
            linewidth=1.6, label="Train loss", color="#2166AC")
    ax.plot(epoch_df["epoch"], epoch_df["val_loss"], marker="s", markersize=4,
            linewidth=1.6, label="Validation loss", color="#B2182B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title)
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_step_loss(step_df, title, outpath, dpi):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(step_df["global_step"], step_df["loss"], linewidth=0.9,
            color="#2166AC", alpha=0.85)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_lr_schedule(step_df, title, outpath, dpi):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(step_df["global_step"], step_df["lr"], linewidth=1.4, color="#1B7837")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.set_title(title)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_stage_comparison(s1_df, s2_df, outpath, dpi):
    """Two-panel side-by-side: Stage 1 vs Stage 2 val loss trajectories.
    Separate panels (not overlaid) since epoch counts and loss ranges differ -
    overlaying would be misleading given they're on different domains."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    axes[0].plot(s1_df["epoch"], s1_df["train_loss"], marker="o", markersize=3,
                 linewidth=1.4, label="Train", color="#2166AC")
    axes[0].plot(s1_df["epoch"], s1_df["val_loss"], marker="s", markersize=3,
                 linewidth=1.4, label="Validation", color="#B2182B")
    axes[0].set_title("Stage 1: Peptide-grammar pretraining")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    axes[1].plot(s2_df["epoch"], s2_df["train_loss"], marker="o", markersize=3,
                 linewidth=1.4, label="Train", color="#2166AC")
    axes[1].plot(s2_df["epoch"], s2_df["val_loss"], marker="s", markersize=3,
                 linewidth=1.4, label="Validation", color="#B2182B")
    axes[1].set_title("Stage 2: AMP fine-tuning")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].legend(frameon=False)
    style_axis(axes[1])

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_log", default="stage1.log")
    ap.add_argument("--stage2_log", default="stage2.log")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.linewidth": 0.8,
        "figure.dpi": 100,  # display only - actual save dpi set separately
    })

    s1_epoch, s1_step = parse_log(args.stage1_log)
    s2_epoch, s2_step = parse_log(args.stage2_log)

    print(f"Stage 1: {len(s1_epoch)} epochs, {len(s1_step)} step records")
    print(f"Stage 2: {len(s2_epoch)} epochs, {len(s2_step)} step records")

    if s1_epoch.empty or s2_epoch.empty:
        raise RuntimeError("Parsed zero epoch-level rows from one or both logs - "
                            "check the log format matches what this script expects.")

    plot_epoch_loss(s1_epoch, "Stage 1: Peptide-grammar pretraining",
                     outdir / "stage1_loss_curve.png", args.dpi)
    plot_epoch_loss(s2_epoch, "Stage 2: AMP fine-tuning",
                     outdir / "stage2_loss_curve.png", args.dpi)

    plot_step_loss(s1_step, "Stage 1: Per-step training loss",
                    outdir / "stage1_step_loss.png", args.dpi)
    plot_step_loss(s2_step, "Stage 2: Per-step training loss",
                    outdir / "stage2_step_loss.png", args.dpi)

    plot_lr_schedule(s1_step, "Stage 1: Learning rate schedule (OneCycleLR)",
                      outdir / "stage1_lr_schedule.png", args.dpi)
    plot_lr_schedule(s2_step, "Stage 2: Learning rate schedule (OneCycleLR)",
                      outdir / "stage2_lr_schedule.png", args.dpi)

    plot_stage_comparison(s1_epoch, s2_epoch,
                           outdir / "stage1_vs_stage2_comparison.png", args.dpi)

    # Save parsed data as CSV too, in case you want to replot elsewhere
    # (e.g. for a manuscript figure editor) without re-parsing logs
    s1_epoch.to_csv(outdir / "stage1_epoch_metrics.csv", index=False)
    s2_epoch.to_csv(outdir / "stage2_epoch_metrics.csv", index=False)
    print(f"\nAll plots and CSVs saved to {outdir}/")


if __name__ == "__main__":
    main()
