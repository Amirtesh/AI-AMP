import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("AMP_features_master.csv")
feature_cols = [c for c in df.columns if c not in ("sequence", "gram_positive", "gram_negative", "fungal")]

print(f"Total rows: {len(df)}, Total feature columns: {len(feature_cols)}")
print(df[feature_cols].describe().T)

# --- 1. Missing values ---
missing = df[feature_cols].isna().sum()
missing = missing[missing > 0]
print("\nColumns with missing values:")
print(missing if len(missing) else "None")

# --- 2. Length distribution - check what you actually have vs typical AMP range ---
plt.figure(figsize=(8, 4))
sns.histplot(df["length"], bins=50)
plt.axvline(50, color="red", linestyle="--", label="typical AMP upper bound (~50 aa)")
plt.legend()
plt.title("Sequence length distribution")
plt.savefig("eda_length_dist.png", dpi=120)
plt.close()
print(f"\nLength stats: min={df['length'].min()}, max={df['length'].max()}, "
      f"median={df['length'].median()}, "
      f"% over 50aa: {100*(df['length']>50).mean():.1f}%")

# --- 3. Outlier detection via IQR on key descriptors, not blind removal ---
def iqr_outlier_mask(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k*iqr, q3 + k*iqr
    return (series < lower) | (series > upper)

key_cols_to_check = ["length", "molecular_weight", "instability_index", "gravy", "charge_at_pH7"]
outlier_summary = {}
for col in key_cols_to_check:
    if col in df.columns:
        mask = iqr_outlier_mask(df[col])
        outlier_summary[col] = mask.sum()
print("\nOutlier counts per key feature (IQR method, informational - do NOT auto-drop yet):")
for col, count in outlier_summary.items():
    print(f"  {col}: {count} ({100*count/len(df):.1f}%)")

# --- 4. Correlation heatmap - check for redundant features before modeling ---
plt.figure(figsize=(14, 12))
corr = df[feature_cols].select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=120)
plt.close()

high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.9:
            high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
print(f"\nHighly correlated feature pairs (|r|>0.9): {len(high_corr_pairs)}")
for a, b, r in high_corr_pairs:
    print(f"  {a} <-> {b}: {r:.3f}")

# --- 5. Class balance ---
print("\nClass balance:")
print(df[["gram_positive", "gram_negative", "fungal"]].mean())

# --- 6. Duplicate/near-duplicate check on features (not just sequence) ---
dupe_feature_rows = df[feature_cols].duplicated().sum()
print(f"\nRows with fully identical feature vectors (different sequences, same descriptors): {dupe_feature_rows}")
