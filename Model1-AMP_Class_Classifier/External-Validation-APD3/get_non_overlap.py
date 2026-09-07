import pandas as pd
import re

def parse_apd3_fasta(path):
    """Handles APD3's export quirk: a banner line masquerading as a FASTA header."""
    records = []
    current_id, current_seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # skip the search-banner pseudo-header explicitly
                if "search led to" in line.lower():
                    continue
                if current_id is not None:
                    records.append((current_id, "".join(current_seq)))
                current_id = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            records.append((current_id, "".join(current_seq)))
    return pd.DataFrame(records, columns=["apd3_id", "sequence"])

gp_apd3 = parse_apd3_fasta("gp_apd3.fasta")
gn_apd3 = parse_apd3_fasta("gn_apd3.fasta")
f_apd3  = parse_apd3_fasta("f_apd3.fasta")

for name, d in [("gram_positive", gp_apd3), ("gram_negative", gn_apd3), ("fungal", f_apd3)]:
    print(f"{name}: {len(d)} records parsed")
    print(d.head(3))
    print()

# --- Normalize sequences the same way as your training pipeline ---
for d in [gp_apd3, gn_apd3, f_apd3]:
    d["sequence"] = d["sequence"].str.upper().str.strip()

# --- Load your existing full training corpus (pre-split, pre-length-filter version) ---
main_df = pd.read_csv("AMP_features_filtered.csv")
existing_sequences = set(main_df["sequence"].str.upper().str.strip())
print(f"Existing training corpus (AMP_features_filtered.csv): {len(existing_sequences)} sequences")

# --- Overlap check per category ---
def check_overlap(apd3_df, category_name):
    apd3_df["in_training_corpus"] = apd3_df["sequence"].isin(existing_sequences)
    n_overlap = apd3_df["in_training_corpus"].sum()
    n_novel = len(apd3_df) - n_overlap
    print(f"\n{category_name}: {len(apd3_df)} total, {n_overlap} already in training corpus "
          f"({100*n_overlap/len(apd3_df):.1f}%), {n_novel} genuinely novel")
    return apd3_df

gp_apd3 = check_overlap(gp_apd3, "gram_positive")
gn_apd3 = check_overlap(gn_apd3, "gram_negative")
f_apd3  = check_overlap(f_apd3, "fungal")

# --- Save only the novel, non-overlapping sequences per category ---
gp_apd3[~gp_apd3["in_training_corpus"]].to_csv("gp_apd3_novel.csv", index=False)
gn_apd3[~gn_apd3["in_training_corpus"]].to_csv("gn_apd3_novel.csv", index=False)
f_apd3[~f_apd3["in_training_corpus"]].to_csv("f_apd3_novel.csv", index=False)

print("\nSaved novel (non-overlapping) subsets: gp_apd3_novel.csv, gn_apd3_novel.csv, f_apd3_novel.csv")
