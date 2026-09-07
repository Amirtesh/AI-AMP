"""
EDA for Model 2 (binary AMP/non-AMP gatekeeper) feature dataset.

Adapted from the Model 1 EDA script, with fixes:
  - Outliers are actually removed (conservative k=3 IQR), not just flagged.
    Both pre- and post-removal files are saved — this is a real modeling
    decision with a real risk (extreme charge/gravy values can be genuine,
    potent AMPs, not errors), so it needs to be reversible and auditable,
    not silent.
  - Description stats, outlier summary, correlation pairs, and the new
    univariate-AUC check are all saved to CSV, not just printed.
  - Length distribution is split by label to verify the length-matching
    fix (from the dataset-construction stage) actually held.
  - NEW: per-feature univariate AUROC against the label. This is the check
    that would have caught the length-leak problem automatically — any
    feature with near-perfect standalone separability is a leak candidate,
    not a good sign.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

INPUT_CSV = "amp_features_model2.csv"
OUTDIR = "eda_model2"

import os
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)
feature_cols = [c for c in df.columns if c not in ("sequence", "label")]
numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"Total rows: {len(df)}, feature columns: {len(feature_cols)} ({len(numeric_feature_cols)} numeric)")

# --- 1. Description stats, saved not just printed ---
desc = df[numeric_feature_cols].describe().T
desc.to_csv(f"{OUTDIR}/description_stats.csv")
print(f"\nSaved {OUTDIR}/description_stats.csv")

# --- 2. Missing values ---
missing = df[feature_cols].isna().sum()
missing = missing[missing > 0]
if len(missing):
    missing.to_csv(f"{OUTDIR}/missing_values.csv")
    print(f"\n[warn] missing values found, saved to {OUTDIR}/missing_values.csv:\n{missing}")
else:
    print("\nNo missing values.")

# --- 3. Length distribution BY LABEL — confirms the length-matching fix held ---
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="length", hue="label", bins=50, element="step", stat="density", common_norm=False)
plt.title("Sequence length distribution by label")
plt.savefig(f"{OUTDIR}/length_dist_by_label.png", dpi=120)
plt.close()

len_by_label = df.groupby("label")["length"].describe()
len_by_label.to_csv(f"{OUTDIR}/length_by_label.csv")
print(f"\nLength by label")
print(len_by_label)

# --- 4. Univariate leak check: per-feature AUROC against label ---
# Any feature that alone predicts the label near-perfectly is a leak
# candidate, the same signature the length problem had before it was caught.
print("\n=== Univariate AUROC per feature (leak check) ===")
auc_results = []
for col in numeric_feature_cols:
    if col == "label":
        continue
    vals = df[col].values
    try:
        auc = roc_auc_score(df["label"], vals)
        auc = max(auc, 1 - auc)  # direction-agnostic — a feature that's a perfect inverse predictor is just as suspicious
        auc_results.append((col, auc))
    except Exception:
        continue

auc_df = pd.DataFrame(auc_results, columns=["feature", "univariate_auc"]).sort_values("univariate_auc", ascending=False)
auc_df.to_csv(f"{OUTDIR}/univariate_auc.csv", index=False)
print(auc_df.head(15).to_string(index=False))

suspicious = auc_df[auc_df["univariate_auc"] > 0.90]
if len(suspicious):
    print(f"\n[FLAG] {len(suspicious)} feature(s) with univariate AUROC > 0.90 — "
          f"investigate before modeling, this is the same signature the length leak had:")
    print(suspicious.to_string(index=False))
else:
    print("\nNo single feature exceeds 0.90 AUROC alone — no obvious leak by this check.")

# --- 5. Outlier detection (conservative k=3) + ACTUAL removal, both files saved ---
def iqr_outlier_mask(series, k=3):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)

# Deliberately conservative and limited to features where an extreme value
# is plausibly a computation artifact rather than real AMP biology. NOT
# applying this to charge/gravy/hydrophobicity-type features, since extreme
# values there can be genuine, potent AMP chemistry — removing them would
# silently bias the dataset toward "average" peptides.
outlier_check_cols = ["length", "molecular_weight", "instability_index"]
outlier_mask_combined = pd.Series(False, index=df.index)
outlier_summary = {}
for col in outlier_check_cols:
    if col in df.columns:
        mask = iqr_outlier_mask(df[col], k=3)
        outlier_summary[col] = int(mask.sum())
        outlier_mask_combined |= mask

pd.Series(outlier_summary, name="outlier_count").to_csv(f"{OUTDIR}/outlier_summary.csv")
print(f"\n=== Outlier removal (k=3 IQR, conservative, only on {outlier_check_cols}) ===")
for col, count in outlier_summary.items():
    print(f"  {col}: {count} ({100*count/len(df):.2f}%)")
print(f"  combined rows flagged for removal: {outlier_mask_combined.sum()} "
      f"({100*outlier_mask_combined.sum()/len(df):.2f}%)")

removed_by_label = df[outlier_mask_combined]["label"].value_counts()
print(f"  removed rows by label (check this isn't skewed toward one class):\n{removed_by_label}")

df_clean = df[~outlier_mask_combined].copy()
df.to_csv(f"{OUTDIR}/../amp_features_model2_with_outliers.csv", index=False)  # explicit backup, unchanged
df_clean.to_csv("amp_features_model2_cleaned.csv", index=False)
print(f"\nSaved amp_features_model2_cleaned.csv ({len(df_clean)} rows, "
      f"{len(df)-len(df_clean)} removed)")
print("Original untouched file remains at amp_features_model2.csv — this step does not overwrite it.")

# --- 6. Correlation heatmap + high-correlation pairs (expect known duplicates) ---
plt.figure(figsize=(16, 14))
corr = df_clean[numeric_feature_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature correlation matrix (post-outlier-removal)")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/correlation_heatmap.png", dpi=120)
plt.close()

high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        r = corr.iloc[i, j]
        if abs(r) > 0.9:
            high_corr_pairs.append((corr.columns[i], corr.columns[j], r))

high_corr_df = pd.DataFrame(high_corr_pairs, columns=["feature_a", "feature_b", "correlation"])
high_corr_df.to_csv(f"{OUTDIR}/high_correlation_pairs.csv", index=False)
print(f"\nHighly correlated pairs (|r|>0.9): {len(high_corr_pairs)} — saved to "
      f"{OUTDIR}/high_correlation_pairs.csv")
print("NOTE: pairs like (length, modlamp_Length), (molecular_weight, modlamp_MW), "
      "(isoelectric_point, modlamp_pI), (instability_index, modlamp_InstabilityInd), "
      "(aromaticity, modlamp_Aromaticity) are EXPECTED — biopython and modlAMP compute "
      "the same underlying quantity independently. This isn't a bug, but it means you "
      "have real duplicate information in the feature set; drop one of each pair before "
      "modeling to avoid diluting feature importance / inflating multicollinearity.")
print(high_corr_df.to_string(index=False))

# --- 7. Class balance ---
print(f"\nClass balance:\n{df_clean['label'].value_counts()}")
print(f"ratio: {(df_clean['label']==0).sum() / (df_clean['label']==1).sum():.3f} : 1")

# --- 8. Duplicate feature vectors (different sequence, identical descriptors) ---
dupe_feature_rows = df_clean[numeric_feature_cols].duplicated().sum()
print(f"\nRows with identical feature vectors despite different sequences: {dupe_feature_rows}")
if dupe_feature_rows > 0:
    print("  Worth a look — could be near-identical short sequences with the same "
          "composition, or a sign that the feature set doesn't distinguish some real "
          "sequence differences. Not necessarily a bug, but not nothing either.")

print(f"\n=== EDA complete. All outputs in {OUTDIR}/ and amp_features_model2_cleaned.csv ===")
