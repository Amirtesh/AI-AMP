import pandas as pd
import glob

def load_sequences(path, seq_col_candidates=("SEQUENCE", "Sequence", "sequence")):
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    seq_col = None
    for c in seq_col_candidates:
        if c in df.columns:
            seq_col = c
            break
    if seq_col is None:
        print(f"WARNING: no known sequence column found in {path}. Columns present: {df.columns.tolist()}")
        raise ValueError(f"Fix seq_col_candidates for {path}")

    seqs = df[seq_col].dropna().astype(str).str.strip().str.upper()
    seqs = seqs[seqs.str.len() > 0]
    print(f"{path}: {len(seqs)} sequences loaded (column '{seq_col}')")
    return set(seqs)

# --- Load every source per category ---
gram_pos_sources = [
    "DBAASP/Antigramp.csv",
    "dbAMP/Antigramp.csv",
    "DRAMP/Anti-Gram-positive_amps.xlsx",
]
gram_neg_sources = [
    "DBAASP/Antigramn.csv",
    "dbAMP/Antigramn.csv",
    "DRAMP/Anti-Gram-_amps.xlsx",
]
fungal_sources = [
    "DBAASP/Antifungal.csv",
    "dbAMP/Antifungal.csv",
    "DRAMP/Antifungal_amps.xlsx",
]

def union_category(sources):
    all_seqs = set()
    for s in sources:
        seqs = load_sequences(s)
        before = len(all_seqs)
        all_seqs |= seqs
        print(f"  -> union so far: {before} + new -> {len(all_seqs)}")
    return all_seqs

print("=== Gram positive ===")
gram_pos_set = union_category(gram_pos_sources)
print("\n=== Gram negative ===")
gram_neg_set = union_category(gram_neg_sources)
print("\n=== Fungal ===")
fungal_set = union_category(fungal_sources)

# --- Build unified multi-label table ---
all_sequences = gram_pos_set | gram_neg_set | fungal_set
print(f"\nTotal unique sequences across all three categories: {len(all_sequences)}")

rows = []
for seq in all_sequences:
    rows.append({
        "SEQUENCE": seq,
        "gram_positive": int(seq in gram_pos_set),
        "gram_negative": int(seq in gram_neg_set),
        "fungal": int(seq in fungal_set),
    })

final_df = pd.DataFrame(rows)

# Sanity checks before saving - don't skip these
print("\nLabel distribution:")
print(final_df[["gram_positive", "gram_negative", "fungal"]].sum())

n_multi_label = (final_df[["gram_positive", "gram_negative", "fungal"]].sum(axis=1) > 1).sum()
print(f"\nSequences active in more than one category (broad-spectrum): {n_multi_label} "
      f"({100*n_multi_label/len(final_df):.1f}%)")

n_zero_label = (final_df[["gram_positive", "gram_negative", "fungal"]].sum(axis=1) == 0).sum()
print(f"Sequences with zero labels (should be 0, investigate if not): {n_zero_label}")

# Duplicate check (should already be handled by set logic, but verify)
dupes = final_df["SEQUENCE"].duplicated().sum()
print(f"Duplicate sequences remaining: {dupes} (should be 0)")

final_df.to_csv("AMP_multilabel_master.csv", index=False)
print(f"\nSaved: AMP_multilabel_master.csv ({len(final_df)} rows)")

