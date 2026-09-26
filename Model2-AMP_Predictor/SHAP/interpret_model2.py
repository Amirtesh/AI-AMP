"""
interpret_model2.py
-------------------
Model 2 (ESM2-only) analogue of interpret_gn.py.

For each of the top-15 SHAP-ranked ESM2 dimensions, reports:
  - the single strongest biophysical correlate (argmax over all biophysical features)
  - the Pearson r for that pair
  - a three-tier classification: meaningful (|r|>=0.5), weak (0.3<=|r|<0.5),
    or no meaningful correlation (<0.3)

Produces the same output shape as gram_negative's summary CSV so the two
tables can be compared side-by-side in Results.

Inputs (same directory):
  AMP2_features_SPLIT_FINAL.csv  — the locked dataset with a 'split' column
  model2_esm2.joblib             — the trained XGBoost joblib model
  feature_columns.json           — {"esm2_cols": [...], "biophysical_cols": [...]}

Outputs (shap_outputs/):
  model2_esm2_correlation_full.csv     — all (dim × biophysical) pairs for top-15
  model2_esm2_correlation_summary.csv  — one best-match row per dim, with interpretation
  model2_esm2_correlation_heatmap.png  — heatmap (mirrors gram_negative version)
"""

import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("shap_outputs", exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("AMP2_features_SPLIT_FINAL.csv")
test_df = df[df["split"] == "test"].reset_index(drop=True)
print(f"Test set: {len(test_df)} rows")

with open("feature_columns.json") as f:
    cols = json.load(f)
esm2_cols        = cols["esm2_cols"]
biophysical_cols = cols["biophysical_cols"]

X_test = test_df[esm2_cols]

# ── 2. Compute SHAP values ────────────────────────────────────────────────────
model     = joblib.load("model2_esm2.joblib")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ── 3. Rank dimensions by mean |SHAP| ────────────────────────────────────────
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_ranking  = pd.DataFrame({
    "esm2_dim":      esm2_cols,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

top_n   = 15
top_dims = shap_ranking.head(top_n)["esm2_dim"].tolist()
print(f"\nTop {top_n} SHAP-important ESM2 dimensions for Model 2:")
print(shap_ranking.head(top_n).to_string(index=False))

# ── 4. Correlate every top dim against every biophysical feature ──────────────
corr_results = []
for dim in top_dims:
    for bio_feat in biophysical_cols:
        r = test_df[dim].corr(test_df[bio_feat])
        corr_results.append({
            "esm2_dim":            dim,
            "shap_rank":           top_dims.index(dim) + 1,
            "biophysical_feature": bio_feat,
            "correlation":         r,
        })

corr_df = pd.DataFrame(corr_results)

# ── 5. Best match per dim (argmax over all biophysical features) ──────────────
best_matches = corr_df.loc[
    corr_df.groupby("esm2_dim")["correlation"].apply(lambda x: x.abs().idxmax())
]
best_matches = best_matches.sort_values("shap_rank").reset_index(drop=True)

print("\n=== Strongest biophysical correlate per top SHAP-ranked ESM2 dimension ===")
print(best_matches.to_string(index=False))

# ── 6. Three-tier classification ─────────────────────────────────────────────
def interpret(r):
    if   abs(r) >= 0.5: return "meaningful correlation - partial interpretability recovered"
    elif abs(r) >= 0.3: return "weak correlation - suggestive only"
    else:               return "no meaningful correlation - genuinely opaque"

best_matches["interpretation"] = best_matches["correlation"].apply(interpret)

print("\n=== Summary classification ===")
print(best_matches[["esm2_dim","shap_rank","biophysical_feature",
                     "correlation","interpretation"]].to_string(index=False))

n_meaningful = (best_matches["interpretation"] == "meaningful correlation - partial interpretability recovered").sum()
n_weak       = (best_matches["interpretation"] == "weak correlation - suggestive only").sum()
n_opaque     = (best_matches["interpretation"] == "no meaningful correlation - genuinely opaque").sum()

print(f"\nOf the top {top_n} SHAP-important dimensions: "
      f"{n_meaningful} meaningfully correlate with a named biophysical property, "
      f"{n_weak} show weak/suggestive correlation, "
      f"{n_opaque} show no meaningful correlation with any biophysical descriptor tested.")

# ── 7. Save CSVs ─────────────────────────────────────────────────────────────
corr_df.to_csv("shap_outputs/model2_esm2_correlation_full.csv",    index=False)
best_matches.to_csv("shap_outputs/model2_esm2_correlation_summary.csv", index=False)

# ── 8. Heatmap (same style as gram_negative) ─────────────────────────────────
pivot_feats  = best_matches["biophysical_feature"].unique().tolist()
heatmap_data = corr_df[corr_df["biophysical_feature"].isin(pivot_feats)].pivot(
    index="esm2_dim", columns="biophysical_feature", values="correlation"
).loc[top_dims]

plt.figure(figsize=(max(8, len(pivot_feats) * 0.8), max(6, len(top_dims) * 0.4)))
plt.imshow(heatmap_data.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=90)
plt.yticks(range(len(heatmap_data.index)),   heatmap_data.index)
plt.colorbar(label="Pearson correlation")
plt.title("Top Model 2 ESM2 dims vs. their best-correlating biophysical features")
plt.tight_layout()
plt.savefig("shap_outputs/model2_esm2_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved: model2_esm2_correlation_full.csv, _summary.csv, _heatmap.png")
