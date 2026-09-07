import pandas as pd
import numpy as np
import json
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("shap_outputs", exist_ok=True)

with open("model_artifacts/config.json") as f:
    config = json.load(f)

test_df = pd.read_csv("test_split.csv")

biophysical_cols = [c for c in test_df.columns if c.startswith("frac_") or c.startswith("modlamp_")
                     or c == "gravy" or c.startswith("ss_")]

# --- Recompute SHAP for gram_negative to get ranked top dimensions ---
feature_cols = config["gram_negative"]["features"]  # esm2 columns
X_test = test_df[feature_cols]

booster = xgb.Booster()
booster.load_model("model_artifacts/gram_negative_model.json")
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X_test)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_ranking = pd.DataFrame({
    "esm2_dim": feature_cols,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

top_n = 15
top_dims = shap_ranking.head(top_n)["esm2_dim"].tolist()
print(f"Top {top_n} SHAP-important ESM2 dimensions for gram_negative:")
print(shap_ranking.head(top_n).to_string(index=False))

# --- Correlate each top dimension against every biophysical descriptor ---
corr_results = []
for dim in top_dims:
    for bio_feat in biophysical_cols:
        r = test_df[dim].corr(test_df[bio_feat])
        corr_results.append({
            "esm2_dim": dim,
            "shap_rank": top_dims.index(dim) + 1,
            "biophysical_feature": bio_feat,
            "correlation": r,
        })

corr_df = pd.DataFrame(corr_results)

# --- For each top ESM2 dim, report its single strongest biophysical correlate ---
best_matches = corr_df.loc[corr_df.groupby("esm2_dim")["correlation"].apply(lambda x: x.abs().idxmax())]
best_matches = best_matches.sort_values("shap_rank").reset_index(drop=True)

print("\n=== Strongest biophysical correlate per top SHAP-ranked ESM2 dimension ===")
print(best_matches.to_string(index=False))

# --- Classify the strength of the mapping, don't just print raw numbers ---
def interpret(r):
    if abs(r) >= 0.5: return "meaningful correlation - partial interpretability recovered"
    elif abs(r) >= 0.3: return "weak correlation - suggestive only"
    else: return "no meaningful correlation - genuinely opaque"

best_matches["interpretation"] = best_matches["correlation"].apply(interpret)
print("\n=== Summary classification ===")
print(best_matches[["esm2_dim","shap_rank","biophysical_feature","correlation","interpretation"]].to_string(index=False))

n_meaningful = (best_matches["interpretation"] == "meaningful correlation - partial interpretability recovered").sum()
n_weak = (best_matches["interpretation"] == "weak correlation - suggestive only").sum()
n_opaque = (best_matches["interpretation"] == "genuinely opaque" ).sum() if False else (best_matches["interpretation"]=="no meaningful correlation - genuinely opaque").sum()

print(f"\nOf the top {top_n} SHAP-important dimensions: "
      f"{n_meaningful} meaningfully correlate with a named biophysical property, "
      f"{n_weak} show weak/suggestive correlation, "
      f"{n_opaque} show no meaningful correlation with any biophysical descriptor tested.")

corr_df.to_csv("shap_outputs/gram_negative_esm2_correlation_full.csv", index=False)
best_matches.to_csv("shap_outputs/gram_negative_esm2_correlation_summary.csv", index=False)

# --- Visual: heatmap of top dims vs top-correlating biophysical features ---
pivot_feats = best_matches["biophysical_feature"].unique().tolist()
heatmap_data = corr_df[corr_df["biophysical_feature"].isin(pivot_feats)].pivot(
    index="esm2_dim", columns="biophysical_feature", values="correlation"
).loc[top_dims]

plt.figure(figsize=(max(8, len(pivot_feats)*0.8), max(6, len(top_dims)*0.4)))
plt.imshow(heatmap_data.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=90)
plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
plt.colorbar(label="Pearson correlation")
plt.title("Top gram_negative ESM2 dims vs. their best-correlating biophysical features")
plt.tight_layout()
plt.savefig("shap_outputs/gram_negative_esm2_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved: gram_negative_esm2_correlation_full.csv, _summary.csv, _heatmap.png")

