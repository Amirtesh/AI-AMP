import pandas as pd

df = pd.read_csv("AMP_features_master.csv")
print(f"Starting rows: {len(df)}")

# --- 1. Exact-identical pairs: drop the Biopython version, keep modlamp ---
exact_dupe_pairs = [
    ("length", "modlamp_Length"),
    ("aromaticity", "modlamp_Aromaticity"),
    ("instability_index", "modlamp_InstabilityInd"),
]
for bio_col, modlamp_col in exact_dupe_pairs:
    diff = df[bio_col].sub(df[modlamp_col]).abs().max()
    print(f"{bio_col} vs {modlamp_col}: max abs diff = {diff}")
    df = df.drop(columns=[bio_col])
    print(f"  -> dropped {bio_col}, kept {modlamp_col}")

# --- 2. Near-identical / consistently-differing pairs: modlamp wins, per your MW finding ---
near_dupe_pairs = [
    ("molecular_weight", "modlamp_MW"),
    ("isoelectric_point", "modlamp_pI"),
    ("charge_at_pH7", "modlamp_Charge"),
]
for bio_col, modlamp_col in near_dupe_pairs:
    diff = df[bio_col].sub(df[modlamp_col]).abs()
    print(f"{bio_col} vs {modlamp_col}: mean abs diff = {diff.mean():.4f}, max abs diff = {diff.max():.4f}")
    df = df.drop(columns=[bio_col])
    print(f"  -> dropped {bio_col}, kept {modlamp_col} (domain-specific tool preferred on disagreement)")

# --- 3. Length filtering (using modlamp_Length, the surviving column) ---
MIN_LEN = 5
MAX_LEN = 60

before = len(df)
too_short = (df["modlamp_Length"] < MIN_LEN).sum()
too_long = (df["modlamp_Length"] > MAX_LEN).sum()
print(f"\nRemoving {too_short} sequences below {MIN_LEN} aa (fragments)")
print(f"Removing {too_long} sequences above {MAX_LEN} aa (small proteins, out of peptide scope)")

df = df[(df["modlamp_Length"] >= MIN_LEN) & (df["modlamp_Length"] <= MAX_LEN)].reset_index(drop=True)
after = len(df)
print(f"\nRows: {before} -> {after} ({before - after} removed, {100*(before-after)/before:.1f}%)")

print("\nClass balance after filtering:")
print(df[["gram_positive", "gram_negative", "fungal"]].mean())

df.to_csv("AMP_features_filtered.csv", index=False)
print(f"\nSaved: AMP_features_filtered.csv ({len(df)} rows, {df.shape[1]} columns)")
print(f"Final columns: {df.columns.tolist()}")
