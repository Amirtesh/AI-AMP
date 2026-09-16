"""
compare_models.py
-----------------
Compares three AMP predictors on a uniform evaluation set defined by the
sequences our ESM-2 model considers valid (valid == True in
predictions_model2_esm2.csv).  All three models are scored on the exact
same sequences so metrics are directly comparable.

Models
------
  1. ESM-2 (ours)  – predictions_model2_esm2.csv   (col: probability)
  2. AMPir         – ampir_predictions.csv           (col: prob_AMP)
  3. AMPEPpy       – predictions_ampeppy.csv         (col: probability_AMP)

Outputs
-------
  benchmark_metrics_full.csv   – metric table
  benchmark_metrics_bar.png    – grouped bar chart
  class_imbalance_bar.png      – AMP vs non-AMP in full dataset and valid subset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

# ── 0. Config ────────────────────────────────────────────────────────────────
THRESHOLD    = 0.5
TRUTH_FILE   = "Cleaned_External_Dataset.csv"
ESM2_FILE    = "predictions_model2_esm2.csv"
AMPIR_FILE   = "ampir_predictions.csv"
AMPEPPY_FILE = "predictions_ampeppy.csv"

# ── 1. Ground truth ──────────────────────────────────────────────────────────
truth = pd.read_csv(TRUTH_FILE).set_index("id_ref")

# ── 2. Determine valid evaluation set from ESM-2 predictions ─────────────────
esm2_raw = pd.read_csv(ESM2_FILE).set_index("id_ref")
valid_ids = esm2_raw[esm2_raw["valid"] == True].index

n_total = len(truth)
n_valid = len(valid_ids)

print(f"Full dataset      : {n_total} sequences")
print(f"Valid (ESM-2)     : {n_valid} sequences")
print(f"Skipped           : {n_total - n_valid} sequences\n")

# Ground truth for the valid subset
y_true = truth.loc[valid_ids, "Label"].values
print(f"Valid subset — AMP: {(y_true == 1).sum()}  |  non-AMP: {(y_true == 0).sum()}\n")

# ── 3. Helper ─────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_score, threshold=THRESHOLD):
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "AUROC"    : roc_auc_score(y_true, y_score),
        "AUPRC"    : average_precision_score(y_true, y_score),
        "F1"       : f1_score(y_true, y_pred, zero_division=0),
        "MCC"      : matthews_corrcoef(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
    }

# ── 4. ESM-2 ─────────────────────────────────────────────────────────────────
y_esm2 = esm2_raw.loc[valid_ids, "probability"].values
assert not np.isnan(y_esm2).any(), "ESM-2: unexpected NaNs in valid set"
metrics_esm2 = compute_metrics(y_true, y_esm2)
print(f"ESM-2 (ours)  — evaluated on {n_valid} sequences")

# ── 5. AMPir ─────────────────────────────────────────────────────────────────
ampir = pd.read_csv(AMPIR_FILE).set_index("id_ref")
y_ampir = ampir.loc[valid_ids, "prob_AMP"].values
assert not np.isnan(y_ampir).any(), "AMPir: unexpected NaNs in valid set"
metrics_ampir = compute_metrics(y_true, y_ampir)
print(f"AMPir         — evaluated on {n_valid} sequences")

# ── 6. AMPEPpy ───────────────────────────────────────────────────────────────
ampeppy = pd.read_csv(AMPEPPY_FILE, sep="\t").set_index("seq_id")
y_ampeppy = ampeppy.loc[valid_ids, "probability_AMP"].values
assert not np.isnan(y_ampeppy).any(), "AMPEPpy: unexpected NaNs in valid set"
metrics_ampeppy = compute_metrics(y_true, y_ampeppy)
print(f"AMPEPpy       — evaluated on {n_valid} sequences")

# ── 8. Results table ─────────────────────────────────────────────────────────
results = pd.DataFrame({
    "ESM-2 (ours)": metrics_esm2,
    "AMPir"       : metrics_ampir,
    "AMPEPpy"     : metrics_ampeppy,
}).T

print("\n── Metrics ──────────────────────────────────────────────────────────")
print(results.to_string(float_format=lambda x: f"{x:.4f}"))
results.to_csv("benchmark_metrics_full.csv")
print("\nSaved: benchmark_metrics_full.csv")

# ── 9. Bar chart – metrics ────────────────────────────────────────────────────
METRIC_COLS = ["AUROC", "AUPRC", "F1", "MCC", "Precision", "Recall"]
MODELS      = results.index.tolist()
N_M         = len(METRIC_COLS)
N_MOD       = len(MODELS)
BAR_W       = 0.24
X           = np.arange(N_M)

COLORS = {
    "ESM-2 (ours)": "#1565C0",
    "AMPir"       : "#546E7A",
    "AMPEPpy"     : "#B0BEC5",
}

fig1, ax1 = plt.subplots(figsize=(13, 5.5))

for i, model in enumerate(MODELS):
    offset = (i - (N_MOD - 1) / 2) * BAR_W
    vals   = results.loc[model, METRIC_COLS].values.astype(float)
    bars   = ax1.bar(
        X + offset, vals,
        width=BAR_W, label=model,
        color=COLORS[model], edgecolor="white", linewidth=0.6, zorder=3,
    )
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, h + 0.011,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=6.5, color="#222222",
            )

ax1.set_xticks(X)
ax1.set_xticklabels(METRIC_COLS, fontsize=11)
ax1.set_ylim(0, 1.16)
ax1.set_ylabel("Score", fontsize=12)
ax1.set_title(
    f"AMP Classifier Comparison  (n = {n_valid} valid sequences)",
    fontsize=13, fontweight="bold", pad=14,
)
ax1.legend(fontsize=10, framealpha=0.9)
ax1.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
ax1.set_axisbelow(True)
ax1.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig1.savefig("benchmark_metrics_bar.png", dpi=180, bbox_inches="tight")
print("Saved: benchmark_metrics_bar.png")
plt.close(fig1)

# ── 10. Bar chart – class imbalance ──────────────────────────────────────────
full_amp    = int((truth["Label"] == 1).sum())
full_nonamp = int((truth["Label"] == 0).sum())
valid_amp   = int((y_true == 1).sum())
valid_nonamp= int((y_true == 0).sum())

categories   = [f"Full Dataset\n(n={n_total})", f"Valid Subset\n(n={n_valid})"]
amp_counts   = [full_amp,    valid_amp]
nonamp_counts= [full_nonamp, valid_nonamp]

x2     = np.arange(len(categories))
bar_w2 = 0.30

fig2, ax2 = plt.subplots(figsize=(7.5, 5))

b1 = ax2.bar(x2 - bar_w2 / 2, amp_counts,    bar_w2,
             label="AMP",     color="#1565C0", edgecolor="white", zorder=3)
b2 = ax2.bar(x2 + bar_w2 / 2, nonamp_counts, bar_w2,
             label="non-AMP", color="#78909C", edgecolor="white", zorder=3)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2, h + 8,
            str(int(h)), ha="center", va="bottom", fontsize=11,
        )

for i, (a, n) in enumerate(zip(amp_counts, nonamp_counts)):
    ax2.text(
        x2[i], max(a, n) + 60,
        f"1 : {n/a:.1f}  (AMP : non-AMP)",
        ha="center", fontsize=9, color="#555555",
    )

ax2.set_xticks(x2)
ax2.set_xticklabels(categories, fontsize=12)
ax2.set_ylabel("Number of Sequences", fontsize=12)
ax2.set_title(
    "Class Distribution — Cleaned External Dataset",
    fontsize=13, fontweight="bold", pad=14,
)
ax2.legend(fontsize=11)
ax2.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_ylim(0, max(full_nonamp, full_amp) * 1.25)

plt.tight_layout()
fig2.savefig("class_imbalance_bar.png", dpi=180, bbox_inches="tight")
print("Saved: class_imbalance_bar.png")
plt.close(fig2)

print("\nDone.")
