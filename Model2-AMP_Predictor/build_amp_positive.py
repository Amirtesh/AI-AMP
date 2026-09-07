#!/usr/bin/env python3
"""
Build a unified AMP positive dataset from three sources:
  DBAASP  -> DBAASP/tier1_confirmed_positives.csv
  dbAMP   -> dbAMP/dbAMP3_pepinfo.xlsx
  DRAMP   -> DRAMP/DRAMP.xlsx

Does NOT assume column names, sheet names, or file structure are exactly as
described — inspects each file first and asserts expected columns exist
before merging. If an assertion fails, the printed diagnostics tell you
what's actually in the file so you can fix the script rather than get a
silently wrong output.

Output: amp_positive.csv with columns:
  sequence          - uppercased, deduplicated
  sequence_length
  n_source_dbs      - how many of the 3 databases contained this sequence
  source_dbs        - e.g. "DBAASP;dbAMP;DRAMP"
  source_ids        - e.g. "DBAASP:1234;dbAMP:dbAMP_00123;DRAMP:DRAMP00456"
  had_lowercase     - True if any raw source sequence had lowercase chars
                       (likely D-amino acid) before uppercasing — check this
                       before feeding to ESM2/ProtParam/modlAMP.

USAGE:
    python build_amp_positive.py
"""

import sys
from pathlib import Path

import pandas as pd

DBAASP_CSV = Path("DBAASP/tier1_confirmed_positives.csv")
DBAMP_XLSX = Path("dbAMP/dbAMP3_pepinfo.xlsx")
DRAMP_XLSX = Path("DRAMP/DRAMP.xlsx")

OUT_CSV = Path("amp_positive.csv")


def fail(msg):
    print(f"\n[FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def load_dbaasp():
    print(f"=== Loading DBAASP: {DBAASP_CSV} ===")
    if not DBAASP_CSV.exists():
        fail(f"{DBAASP_CSV} not found. Run this from AMP-Classification/ directory.")
    df = pd.read_csv(DBAASP_CSV)
    print(f"  shape: {df.shape}, columns: {list(df.columns)}")
    expected = {"peptideId", "sequence"}
    missing = expected - set(df.columns)
    if missing:
        fail(f"DBAASP file missing expected columns: {missing}. Actual columns: {list(df.columns)}")

    df = df.rename(columns={"peptideId": "source_id"})
    df["source_db"] = "DBAASP"
    df["source_id"] = "DBAASP:" + df["source_id"].astype(str)
    return df[["source_db", "source_id", "sequence"]]


def load_dbamp():
    print(f"\n=== Loading dbAMP: {DBAMP_XLSX} ===")
    if not DBAMP_XLSX.exists():
        fail(f"{DBAMP_XLSX} not found.")

    xl = pd.ExcelFile(DBAMP_XLSX)
    print(f"  sheets found: {xl.sheet_names}")
    if len(xl.sheet_names) > 1:
        print(f"  [warn] multiple sheets — using first ('{xl.sheet_names[0]}'). "
              f"Confirm this is the right one, not a summary/index sheet.")
    df = xl.parse(xl.sheet_names[0])
    print(f"  shape: {df.shape}, columns: {list(df.columns)}")

    expected = {"dbAMP_ID", "Seq"}
    missing = expected - set(df.columns)
    if missing:
        fail(f"dbAMP file missing expected columns: {missing}. Actual columns: {list(df.columns)}")

    df = df.rename(columns={"dbAMP_ID": "source_id", "Seq": "sequence"})
    df["source_db"] = "dbAMP"
    df["source_id"] = "dbAMP:" + df["source_id"].astype(str)
    return df[["source_db", "source_id", "sequence"]]


def load_dramp():
    print(f"\n=== Loading DRAMP: {DRAMP_XLSX} ===")
    if not DRAMP_XLSX.exists():
        fail(f"{DRAMP_XLSX} not found.")

    xl = pd.ExcelFile(DRAMP_XLSX)
    print(f"  sheets found: {xl.sheet_names}")
    if len(xl.sheet_names) > 1:
        print(f"  [warn] multiple sheets — using first ('{xl.sheet_names[0]}'). "
              f"Confirm this is the right one.")
    df = xl.parse(xl.sheet_names[0])
    print(f"  shape: {df.shape}, columns: {list(df.columns)}")

    expected = {"DRAMP_ID", "Sequence"}
    missing = expected - set(df.columns)
    if missing:
        fail(f"DRAMP file missing expected columns: {missing}. Actual columns: {list(df.columns)}")

    df = df.rename(columns={"DRAMP_ID": "source_id", "Sequence": "sequence"})
    df["source_db"] = "DRAMP"
    df["source_id"] = "DRAMP:" + df["source_id"].astype(str)
    return df[["source_db", "source_id", "sequence"]]


def main():
    dbaasp = load_dbaasp()
    dbamp = load_dbamp()
    dramp = load_dramp()

    combined = pd.concat([dbaasp, dbamp, dramp], ignore_index=True)
    print(f"\n=== Combined raw rows: {len(combined)} ===")

    # --- Basic sanity checks before dedup, not after ---
    before = len(combined)
    combined["sequence"] = combined["sequence"].astype(str).str.strip()
    empty_mask = (combined["sequence"] == "") | (combined["sequence"].str.lower() == "nan")
    n_empty = empty_mask.sum()
    if n_empty:
        print(f"  [warn] dropping {n_empty} rows with empty/NaN sequence")
    combined = combined[~empty_mask].copy()

    # non-standard-character check — flag, don't silently strip. Real peptide
    # sequences can contain unusual residues; don't assume every non-ACDEFGHIKLMNPQRSTVWY
    # character is junk. Just surface it.
    standard_aa = set("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy")
    def has_nonstandard(seq):
        return any(c not in standard_aa for c in seq)
    combined["has_nonstandard_char"] = combined["sequence"].apply(has_nonstandard)
    n_nonstd = combined["has_nonstandard_char"].sum()
    if n_nonstd:
        sample = combined[combined["has_nonstandard_char"]]["sequence"].head(5).tolist()
        print(f"  [flag] {n_nonstd} sequences contain characters outside the standard 20 AA "
              f"(upper or lower). NOT auto-removed — inspect before deciding. Sample: {sample}")

    # --- Case tracking and normalization ---
    combined["had_lowercase"] = combined["sequence"].str.contains(r"[a-z]", regex=True)
    combined["sequence"] = combined["sequence"].str.upper()
    combined["sequence_length"] = combined["sequence"].str.len()

    n_lower = combined["had_lowercase"].sum()
    print(f"\n  {n_lower} / {len(combined)} raw entries had lowercase (D-amino acid) characters, "
          f"now uppercased per your explicit instruction. This discards stereochemistry "
          f"information — confirmed as your deliberate call, not assumed by this script.")

    # --- Dedup on uppercased sequence, preserving provenance ---
    print(f"\n=== Deduplicating on uppercased sequence ===")
    grouped = combined.groupby("sequence").agg(
        sequence_length=("sequence_length", "first"),
        source_dbs=("source_db", lambda x: ";".join(sorted(set(x)))),
        n_source_dbs=("source_db", lambda x: len(set(x))),
        source_ids=("source_id", lambda x: ";".join(x)),
        had_lowercase=("had_lowercase", "any"),
        has_nonstandard_char=("has_nonstandard_char", "any"),
    ).reset_index()

    print(f"  unique sequences after dedup: {len(grouped)} (from {len(combined)} raw rows)")
    overlap_dist = grouped["n_source_dbs"].value_counts().sort_index()
    print(f"  cross-database overlap distribution:\n{overlap_dist}")
    pct_shared = (grouped['n_source_dbs'] > 1).sum() / len(grouped) * 100
    print(f"  {pct_shared:.1f}% of unique sequences appear in more than one database "
          f"(compare against your known ~72% cross-database duplication figure from Model 1 — "
          f"if this is wildly different, that's worth understanding before proceeding, not ignoring).")

    # --- Remove non-standard-amino-acid-containing peptides (explicit instruction) ---
    n_before_filter = len(grouped)
    n_removed = int(grouped["has_nonstandard_char"].sum())
    removed_examples = grouped[grouped["has_nonstandard_char"]][["sequence", "source_dbs"]].head(10)
    grouped = grouped[~grouped["has_nonstandard_char"]].copy()
    print(f"\n=== Removing non-standard-amino-acid-containing peptides ===")
    print(f"  removed {n_removed} / {n_before_filter} sequences "
          f"({n_removed / n_before_filter * 100:.2f}%)")
    if n_removed:
        print(f"  sample removed sequences (for your own sanity check, not just trusting the filter):")
        print(removed_examples.to_string(index=False))
    print(f"  remaining: {len(grouped)}")

    grouped.to_csv(OUT_CSV, index=False)
    print(f"\n=== Wrote {OUT_CSV} ({len(grouped)} rows) ===")


if __name__ == "__main__":
    main()
