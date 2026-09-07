#!/usr/bin/env python3
"""
Apply a single, consistent length filter (5-100 amino acids) to both:
  - amp_positive.csv          (positive class)
  - uniprot_background_raw.tsv (negative-class candidate pool)

Range chosen based on literature convergence, not an arbitrary round number:
  - ampir: mature AMPs typically 10-50 aa
  - HMAMP: training data capped at 52 aa
  - OmegAMP: capped at 100 aa (explicit AMP-generation precedent for the
    upper bound used here)
  - AMPScanner: flags <10aa predictions as low-confidence (informs the
    floor, though we're using 5 as the hard cutoff since below that is
    biophysically implausible for AMP activity, not just low-confidence)
  - Confirmed empirically against your own data: sequences <5aa in
    amp_positive.csv are single/poly-residue artifacts (e.g. "A", "AAAA"),
    not real peptides, concentrated disproportionately in dbAMP-only entries.

Writes filtered outputs alongside the originals rather than overwriting,
so you can compare before/after and roll back if the range needs revisiting.
"""

import sys
from pathlib import Path

import pandas as pd

MIN_LEN = 5
MAX_LEN = 100

POS_IN = Path("amp_positive.csv")
POS_OUT = Path("amp_positive_filtered.csv")

BG_IN = Path("uniprot_background_raw.tsv")
BG_OUT = Path("uniprot_background_filtered.tsv")


def filter_positives():
    print(f"=== Filtering {POS_IN} ===")
    if not POS_IN.exists():
        print(f"[skip] {POS_IN} not found", file=sys.stderr)
        return
    df = pd.read_csv(POS_IN)
    before = len(df)
    mask = df["sequence_length"].between(MIN_LEN, MAX_LEN)
    kept = df[mask].copy()
    print(f"  before: {before}")
    print(f"  removed <{MIN_LEN}aa: {(df['sequence_length'] < MIN_LEN).sum()}")
    print(f"  removed >{MAX_LEN}aa: {(df['sequence_length'] > MAX_LEN).sum()}")
    print(f"  kept: {len(kept)} ({len(kept)/before*100:.1f}%)")
    kept.to_csv(POS_OUT, index=False)
    print(f"  wrote {POS_OUT}")


def filter_background():
    print(f"\n=== Filtering {BG_IN} ===")
    if not BG_IN.exists():
        print(f"[skip] {BG_IN} not found", file=sys.stderr)
        return
    df = pd.read_csv(BG_IN, sep="\t")
    if "Length" not in df.columns:
        print(f"[error] expected column 'Length' not found. Actual columns: {list(df.columns)}",
              file=sys.stderr)
        sys.exit(1)
    before = len(df)
    mask = df["Length"].between(MIN_LEN, MAX_LEN)
    kept = df[mask].copy()
    print(f"  before: {before}")
    print(f"  removed <{MIN_LEN}aa: {(df['Length'] < MIN_LEN).sum()}")
    print(f"  removed >{MAX_LEN}aa: {(df['Length'] > MAX_LEN).sum()}")
    print(f"  kept: {len(kept)} ({len(kept)/before*100:.1f}%)")
    kept.to_csv(BG_OUT, sep="\t", index=False)
    print(f"  wrote {BG_OUT}")


if __name__ == "__main__":
    filter_positives()
    filter_background()
