# pip install modlamp biopython

"""
Feature extraction for Model 2 (binary AMP/non-AMP gatekeeper).

Adapted from the Model 1 feature-extraction script, with fixes:
  - modlAMP descriptor column order is VERIFIED via --test before trusting
    it across the full run (the original script left this as an unverified
    assumption — "print featurenames to confirm before trusting the zip" —
    that was never actually acted on as far as I can confirm from you).
  - frac_* columns use sorted(STANDARD_AA), not raw set iteration, for
    deterministic column order across runs (Python string hashing is
    randomized per-process by default).
  - Exceptions are logged with type/message, not silently swallowed —
    dropped counts alone don't tell you if there's a systematic bug.
  - Progress printed every N rows; incremental checkpointed writes so a
    68k-row run surviving a crash doesn't mean starting over.

USAGE:
    python3 create_feature_dataset_model2.py --test          # first 5 sequences only, verify modlAMP alignment
    python3 create_feature_dataset_model2.py                 # full run
    python3 create_feature_dataset_model2.py --resume         # continue an interrupted run
"""

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from modlamp.descriptors import GlobalDescriptor

STANDARD_AA = sorted("ACDEFGHIKLMNPQRSTVWY")  # sorted -> deterministic column order

INPUT_CSV = "amp_dataset_model2_v1_prelim.csv"
SEQ_COL = "sequence"
LABEL_COL = "label"
OUTPUT_CSV = "amp_features_model2.csv"
DONE_IDS_FILE = "feature_extraction_done_indices.txt"


def is_valid_sequence(seq):
    return len(seq) > 0 and set(seq.upper()) <= set(STANDARD_AA)


def compute_features(seq, verbose_errors=False):
    seq = seq.upper()
    features = {"sequence": seq, "length": len(seq)}

    for aa in STANDARD_AA:
        features[f"frac_{aa}"] = seq.count(aa) / len(seq)

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
        if verbose_errors:
            print(f"  [ProtParam error] seq={seq[:30]}... {type(e).__name__}: {e}", file=sys.stderr)
        return None, f"ProtParam:{type(e).__name__}"

    try:
        desc = GlobalDescriptor(seq)
        desc.calculate_all()
        values = desc.descriptor[0]
        names = desc.featurenames
        for name, val in zip(names, values):
            features[f"modlamp_{name}"] = val
    except Exception as e:
        if verbose_errors:
            print(f"  [modlAMP error] seq={seq[:30]}... {type(e).__name__}: {e}", file=sys.stderr)
        return None, f"modlAMP:{type(e).__name__}"

    return features, None


def process_chunk(chunk):
    """Runs in a worker process. Pure compute, no I/O — writing happens only
    in the main process, to keep the single-writer checkpoint guarantee."""
    results = []
    for idx, seq, label in chunk:
        seq = str(seq)
        if not is_valid_sequence(seq):
            results.append((idx, None, "invalid_chars"))
            continue
        feats, err = compute_features(seq)
        if feats is None:
            results.append((idx, None, err))
            continue
        feats[LABEL_COL] = label
        results.append((idx, feats, None))
    return results


def run_test():
    print("=== TEST MODE: verifying modlAMP column alignment on 5 sequences ===")
    df = pd.read_csv(INPUT_CSV)
    sample = df[SEQ_COL].head(5).tolist()
    for seq in sample:
        seq = str(seq).upper()
        if not is_valid_sequence(seq):
            print(f"  [skip, non-standard] {seq}")
            continue
        desc = GlobalDescriptor(seq)
        desc.calculate_all()
        names = desc.featurenames
        values = desc.descriptor[0]
        print(f"\nseq: {seq}")
        print(f"  n names: {len(names)}, n values: {len(values)}")
        if len(names) != len(values):
            print(f"  [MISMATCH] names and values have different lengths — "
                  f"do NOT trust zip(names, values). Investigate modlAMP version/API before proceeding.",
                  file=sys.stderr)
            sys.exit(1)
        for n, v in zip(names, values):
            print(f"    {n}: {v}")
    print("\nIf the names above look like sensible descriptor labels (not generic "
          "'feature_0', 'feature_1' placeholders) and match known modlAMP GlobalDescriptor "
          "outputs (e.g. Length, MW, Charge, pI, InstabilityIndex, Aromaticity, "
          "AliphaticIndex, BomanIndex, HydrophobicRatio), the alignment is trustworthy. "
          "If anything looks off, stop and investigate before running the full extraction.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8,
                     help="Process pool size. Defaults to 8 (physical-core count on an "
                          "8c/16t chip) rather than logical thread count — CPU-bound work "
                          "like this typically doesn't benefit from SMT/hyperthreading the "
                          "way I/O-bound work does. Override and compare empirically if curious.")
    args = ap.parse_args()

    if args.test:
        run_test()
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Total input sequences: {len(df)}")

    done_indices = set()
    if args.resume and Path(DONE_IDS_FILE).exists():
        done_indices = {int(x) for x in Path(DONE_IDS_FILE).read_text().split() if x.strip()}
        print(f"--resume: skipping {len(done_indices)} already-processed rows")

    write_header = not (Path(OUTPUT_CSV).exists() and args.resume)
    out_mode = "a" if (args.resume and Path(OUTPUT_CSV).exists()) else "w"
    done_file = open(DONE_IDS_FILE, "a" if args.resume else "w")

    n_ok = 0
    n_dropped = 0
    error_counts = {}
    first_write = write_header
    buffer = []
    BUFFER_SIZE = 200
    CHUNK_SIZE = 100

    def flush_buffer():
        nonlocal first_write
        if not buffer:
            return
        chunk_df = pd.DataFrame(buffer)
        chunk_df.to_csv(OUTPUT_CSV, mode="a" if not first_write else out_mode,
                         header=first_write, index=False)
        first_write = False
        buffer.clear()

    work_items = [
        (idx, str(getattr(row, SEQ_COL)), getattr(row, LABEL_COL))
        for idx, row in enumerate(df.itertuples(index=False))
        if idx not in done_indices
    ]
    print(f"Rows to process: {len(work_items)} (skipped {len(done_indices)} already done)")

    chunks = [work_items[i:i + CHUNK_SIZE] for i in range(0, len(work_items), CHUNK_SIZE)]
    n_processed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_chunk, c) for c in chunks]
        for future in as_completed(futures):
            chunk_results = future.result()
            for idx, feats, err in chunk_results:
                n_processed += 1
                if feats is None:
                    n_dropped += 1
                    error_counts[err] = error_counts.get(err, 0) + 1
                    done_file.write(f"{idx}\n")
                    continue
                buffer.append(feats)
                n_ok += 1
                done_file.write(f"{idx}\n")

                if len(buffer) >= BUFFER_SIZE:
                    flush_buffer()
                    done_file.flush()

            if n_processed % args.progress_every < CHUNK_SIZE:
                print(f"  [{n_processed}/{len(work_items)}] ok={n_ok} dropped={n_dropped}")

    flush_buffer()
    done_file.close()

    print(f"\nDone.")
    print(f"Successfully featurized: {n_ok}")
    print(f"Dropped: {n_dropped}")
    if error_counts:
        print("Drop reasons:")
        for reason, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
