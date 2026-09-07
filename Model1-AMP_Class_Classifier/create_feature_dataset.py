# pip install modlamp biopython

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from modlamp.descriptors import GlobalDescriptor

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_sequence(seq):
    return len(seq) > 0 and set(seq.upper()) <= STANDARD_AA

def compute_features(seq):
    seq = seq.upper()
    features = {"sequence": seq, "length": len(seq)}

    # --- Amino acid composition (string-based) ---
    for aa in STANDARD_AA:
        features[f"frac_{aa}"] = seq.count(aa) / len(seq)

    # --- Biopython ProtParam ---
    try:
        pa = ProteinAnalysis(seq)
        features["molecular_weight"] = pa.molecular_weight()
        features["aromaticity"] = pa.aromaticity()
        features["instability_index"] = pa.instability_index()
        features["isoelectric_point"] = pa.isoelectric_point()
        features["gravy"] = pa.gravy()
        helix, turn, sheet = pa.secondary_structure_fraction()
        features["ss_helix_frac"] = helix
        features["ss_turn_frac"] = turn
        features["ss_sheet_frac"] = sheet
        features["charge_at_pH7"] = pa.charge_at_pH(7.0)
    except Exception as e:
        return None  # fails ProtParam parsing - not a normal sequence, drop it

    # --- modlAMP AMP-specific descriptors ---
    try:
        desc = GlobalDescriptor(seq)
        desc.calculate_all()
        # column order per modlAMP GlobalDescriptor.calculate_all() docs -
        # PRINT desc.featurenames the first time you run this to confirm order/names
        # before trusting the zip below
        values = desc.descriptor[0]
        names = desc.featurenames
        for name, val in zip(names, values):
            features[f"modlamp_{name}"] = val
    except Exception as e:
        return None  # modlAMP failed to parse - drop, not a normal peptide

    return features

# --- Run over master dataset ---
df = pd.read_csv("AMP_multilabel_master.csv")

rows = []
dropped = []
for _, row in df.iterrows():
    seq = row["SEQUENCE"]
    if not is_valid_sequence(seq):
        dropped.append(seq)
        continue
    feats = compute_features(seq)
    if feats is None:
        dropped.append(seq)
        continue
    feats["gram_positive"] = row["gram_positive"]
    feats["gram_negative"] = row["gram_negative"]
    feats["fungal"] = row["fungal"]
    rows.append(feats)

feature_df = pd.DataFrame(rows)
print(f"Total input sequences: {len(df)}")
print(f"Successfully featurized: {len(feature_df)}")
print(f"Dropped (non-standard residues or parsing failure): {len(dropped)}")
print(f"Dropped fraction: {100*len(dropped)/len(df):.1f}%")

feature_df.to_csv("AMP_features_master.csv", index=False)

