#!/usr/bin/env python3
"""
Build the final length-matched Model 2 dataset.

Fixes the length-distribution mismatch found in amp_dataset_final.csv:
  positive median length 20 (IQR 13-35), negative median length 75 (IQR 57-89)
  — background was overwhelmingly long-protein-derived while AMPs are
  overwhelmingly short, creating a trivial length-based signal a model could
  exploit instead of learning real sequence chemistry.

Strategy: match background to the positive class at EXACT length (not coarse
bins), for the tightest possible distribution match:
  - Where background naturally has enough entries at a given length: sample
    directly (real, non-fragmented sequences — preferred whenever possible).
  - Where background is short (mainly the 5-30aa range, where positives are
    concentrated but few natural short proteins exist): generate fragments
    of the exact needed length from the surplus of unused long background
    proteins (mainly the 60-100aa range, which has far more entries than
    needed for direct matching).

Fixed random seed for reproducibility — same principle already applied to
the CD-HIT split in Model 1: this dataset composition must be re-derivable,
not regenerated differently each run.

USAGE:
    python3 build_length_matched_dataset.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

POS_IN = Path("amp_positive_filtered.csv")
BG_IN = Path("uniprot_background_filtered.tsv")
OUT_CSV = Path("amp_dataset_length_matched.csv")

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 5, 100


def fail(msg):
    print(f"\n[FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def load_positives():
    if not POS_IN.exists():
        fail(f"{POS_IN} not found.")
    df = pd.read_csv(POS_IN)
    df["label"] = 1
    df = df.rename(columns={"source_dbs": "source_db", "source_ids": "source_id"})
    return df[["sequence", "sequence_length", "label", "source_db", "source_id"]]


def load_clean_background(positive_sequences):
    if not BG_IN.exists():
        fail(f"{BG_IN} not found.")
    df = pd.read_csv(BG_IN, sep="\t")
    df = df.rename(columns={"Entry": "source_id", "Length": "sequence_length", "Sequence": "sequence"})
    df["sequence"] = df["sequence"].astype(str).str.strip()

    has_nonstd = df["sequence"].apply(lambda s: any(c not in STANDARD_AA for c in s))
    df = df[~has_nonstd].copy()

    overlap_mask = df["sequence"].isin(positive_sequences)
    df = df[~overlap_mask].copy()

    df = df.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)
    df["label"] = 0
    df["source_db"] = "UniProt"
    return df


def main():
    positives = load_positives()
    print(f"Positives: {len(positives)}")

    background = load_clean_background(set(positives["sequence"]))
    print(f"Background (clean, deduped): {len(background)}")

    target_counts = positives["sequence_length"].value_counts().to_dict()

    direct_matched = []
    used_bg_indices = set()
    deficits = {}  # length -> how many more we need

    print("\n=== Pass 1: direct exact-length matching ===")
    for length, n_needed in sorted(target_counts.items()):
        candidates = background[
            (background["sequence_length"] == length) & (~background.index.isin(used_bg_indices))
        ]
        n_available = len(candidates)
        n_take = min(n_needed, n_available)
        if n_take > 0:
            sampled = candidates.sample(n=n_take, random_state=SEED)
            direct_matched.append(sampled)
            used_bg_indices.update(sampled.index)
        if n_available < n_needed:
            deficits[length] = n_needed - n_available

    direct_df = pd.concat(direct_matched, ignore_index=True) if direct_matched else pd.DataFrame()
    print(f"  directly matched: {len(direct_df)}")
    total_deficit = sum(deficits.values())
    print(f"  total deficit needing fragments: {total_deficit} across {len(deficits)} lengths")

    print("\n=== Pass 2: fragmenting surplus long proteins to fill deficits ===")
    # Surplus pool: unused background entries long enough to be cut down to
    # at least the largest deficit length. Prefer the longest unused entries
    # first so we're drawing from genuine surplus, not entries that were
    # borderline-useful for direct matching elsewhere.
    surplus_pool = background[~background.index.isin(used_bg_indices)].copy()
    surplus_pool = surplus_pool.sort_values("sequence_length", ascending=False)
    print(f"  surplus pool available: {len(surplus_pool)}")

    fragments = []
    surplus_list = surplus_pool.to_dict("records")
    surplus_ptr = 0

    for length in sorted(deficits.keys(), reverse=True):  # fill largest-length deficits first
        n_needed = deficits[length]
        n_made = 0
        attempts = 0
        max_attempts = n_needed * 20 + 100  # generous cap so this can't spin forever
        while n_made < n_needed and attempts < max_attempts and surplus_ptr < len(surplus_list) * 3:
            attempts += 1
            src = surplus_list[surplus_ptr % len(surplus_list)]
            surplus_ptr += 1
            src_len = src["sequence_length"]
            if src_len < length:
                continue
            max_start = src_len - length
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            frag_seq = src["sequence"][start:start + length]
            if len(frag_seq) != length:
                continue
            fragments.append({
                "sequence": frag_seq,
                "sequence_length": length,
                "label": 0,
                "source_db": "UniProt_fragment",
                "source_id": f"{src['source_id']}:frag{start}-{start+length}",
            })
            n_made += 1
        if n_made < n_needed:
            print(f"  [warn] length {length}: only generated {n_made}/{n_needed} fragments "
                  f"(ran out of usable surplus source material)", file=sys.stderr)

    frag_df = pd.DataFrame(fragments)
    print(f"  fragments generated: {len(frag_df)}")

    # Dedup fragments against themselves, against direct-matched background,
    # and against positives (a random fragment coincidentally matching a real
    # AMP is unlikely but not impossible at short lengths — check, don't assume)
    if len(frag_df):
        frag_df = frag_df.drop_duplicates(subset="sequence", keep="first")
        frag_df = frag_df[~frag_df["sequence"].isin(direct_df["sequence"] if len(direct_df) else [])]
        frag_df = frag_df[~frag_df["sequence"].isin(positives["sequence"])]
        print(f"  fragments after dedup: {len(frag_df)}")

    negative_final = pd.concat(
        [direct_df[["sequence", "sequence_length", "label", "source_db", "source_id"]] if len(direct_df) else pd.DataFrame(),
         frag_df[["sequence", "sequence_length", "label", "source_db", "source_id"]] if len(frag_df) else pd.DataFrame()],
        ignore_index=True
    )
    negative_final["label"] = negative_final["label"].astype(int)

    print(f"\n=== Merging ===")
    final = pd.concat([positives, negative_final], ignore_index=True)

    label_counts_per_seq = final.groupby("sequence")["label"].nunique()
    conflicted = label_counts_per_seq[label_counts_per_seq > 1]
    if len(conflicted):
        fail(f"{len(conflicted)} sequences appear with BOTH labels — do not proceed. "
             f"Examples: {conflicted.index[:5].tolist()}")

    final.to_csv(OUT_CSV, index=False)

    print(f"\n=== Wrote {OUT_CSV} ===")
    print(f"  total: {len(final)}")
    print(f"  label=1 (AMP):     {(final['label']==1).sum()}")
    print(f"  label=0 (non-AMP): {(final['label']==0).sum()}")
    print(f"    of which fragments: {(final['source_db']=='UniProt_fragment').sum()}")
    print(f"\n  sequence_length by label (should now be close between the two):")
    print(final.groupby("label")["sequence_length"].describe())


if __name__ == "__main__":
    main()
