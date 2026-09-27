# pip install shap

import shap
import xgboost as xgb
import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")  # save-to-file, no display backend needed
import matplotlib.pyplot as plt

os.makedirs("shap_outputs", exist_ok=True)

with open("model_artifacts/config.json") as f:
    config = json.load(f)

test_df = pd.read_csv("test_split.csv")

for task in ["gram_positive", "gram_negative", "fungal"]:
    print(f"\n=== SHAP: {task} ({config[task]['feature_set_name']}) ===")

    feature_cols = config[task]["features"]
    X_test = test_df[feature_cols]

    booster = xgb.Booster()
    booster.load_model(f"model_artifacts/{task}_model.json")

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_test)

    # --- 1. Bar plot: mean |SHAP| per feature, top 20 ---
    plt.figure(figsize=(14, 7))
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=20, show=False)
    plt.title(f"{task} - Feature Importance (mean |SHAP|)")
    ax = plt.gca()
    # SHAP sets a long x-axis label; re-apply it with wrapping so it never clips
    current_xlabel = ax.get_xlabel()
    ax.set_xlabel(current_xlabel, wrap=True, labelpad=10)
    plt.savefig(f"shap_outputs/{task}_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 2. Beeswarm: distribution of SHAP values per feature ---
    plt.figure()
    shap.summary_plot(shap_values, X_test, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(f"shap_outputs/{task}_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 3. Waterfall plots for individual predictions (replaces broken force plots) ---
    for i in [0, 1, 2]:
        exp = shap.Explanation(
            values=shap_values[i],
            base_values=explainer.expected_value,
            data=X_test.iloc[i].values,
            feature_names=X_test.columns.tolist()
        )
        plt.figure()
        shap.plots.waterfall(exp, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(f"shap_outputs/{task}_waterfall_example{i}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved: {task}_bar.png, {task}_beeswarm.png, {task}_waterfall_example[0-2].png")

print("\nAll SHAP outputs saved to shap_outputs/")
