# pip install shap

import shap
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("shap_outputs", exist_ok=True)

# --- Load the actual locked test split (has biophysical + ESM2 + split columns together) ---
df = pd.read_csv("AMP2_features_SPLIT_FINAL.csv")
test_df = df[df["split"] == "test"].reset_index(drop=True)
print(f"Test set: {len(test_df)} rows")

with open("feature_columns.json") as f:
    cols = json.load(f)
esm2_cols = cols["esm2_cols"]
biophysical_cols = cols["biophysical_cols"]

X_test = test_df[esm2_cols]

model = joblib.load("model2_esm2.joblib")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# --- 1. Bar plot ---
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=20, show=False)
plt.title("Model 2 (ESM2-only) - Feature Importance (mean |SHAP|)")
plt.tight_layout()
plt.savefig("shap_outputs/model2_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 2. Beeswarm ---
plt.figure()
shap.summary_plot(shap_values, X_test, max_display=20, show=False)
plt.tight_layout()
plt.savefig("shap_outputs/model2_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 3. Waterfall plots: reasoned example selection, not positional [0,1,2] ---
probs = model.predict_proba(X_test)[:, 1]
test_df["pred_prob"] = probs
test_df["true_label"] = test_df["label"]

correct_amp_idx = test_df[test_df["true_label"] == 1]["pred_prob"].idxmax()
correct_nonamp_idx = test_df[test_df["true_label"] == 0]["pred_prob"].idxmin()

test_df["is_wrong"] = ((test_df["pred_prob"] >= 0.5).astype(int) != test_df["true_label"])
wrong_df = test_df[test_df["is_wrong"]].copy()

example_indices = [correct_amp_idx, correct_nonamp_idx]
example_labels = ["correct_AMP", "correct_nonAMP"]

if len(wrong_df):
    wrong_df["confidence"] = np.abs(wrong_df["pred_prob"] - 0.5)
    most_wrong_idx = wrong_df["confidence"].idxmax()
    example_indices.append(most_wrong_idx)
    example_labels.append("misclassified")
else:
    print("No misclassifications in test set — unusual, worth double-checking before assuming it's simply a great model")

print("\nSelected examples:")
print(test_df.loc[example_indices, ["sequence", "true_label", "pred_prob"]].to_string())

for idx, label_name in zip(example_indices, example_labels):
    row_pos = test_df.index.get_loc(idx)
    exp = shap.Explanation(
        values=shap_values[row_pos],
        base_values=explainer.expected_value,
        data=X_test.iloc[row_pos].values,
        feature_names=X_test.columns.tolist()
    )
    plt.figure()
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(f"shap_outputs/model2_waterfall_{label_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved model2_waterfall_{label_name}.png "
          f"(true_label={test_df.loc[idx,'true_label']}, pred_prob={test_df.loc[idx,'pred_prob']:.4f})")

# --- 4. Correlation probe: what do the top anonymous ESM2 dims actually track? ---
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_dims_idx = np.argsort(mean_abs_shap)[::-1][:15]
top_dims = [esm2_cols[i] for i in top_dims_idx]

probe_results = []
for dim in top_dims:
    for bio_col in biophysical_cols:
        r = np.corrcoef(test_df[dim], test_df[bio_col])[0, 1]
        if abs(r) > 0.5:
            probe_results.append({"esm2_dim": dim, "biophysical_feature": bio_col, "correlation": r})

probe_df = pd.DataFrame(probe_results).sort_values("correlation", key=abs, ascending=False)
probe_df.to_csv("shap_outputs/model2_esm2_correlation_probe.csv", index=False)
print(f"\n{len(set(probe_df['esm2_dim']))} / 15 top dims had at least one |r|>0.5 correlation with a named biophysical feature")
print(probe_df.to_string(index=False))

print("\nAll SHAP outputs saved to shap_outputs/")
