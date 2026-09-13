"""
hemopi2_client.py
In-process HemoPI2 Hybrid2 (ESM + MERCI) hemolysis scoring, matching
`hemopi2_classification -m 4` output exactly, by calling the real
installed package functions directly (not a reimplementation) — this
avoids silently diverging from the validated CLI numbers (v4: 75.7%,
Case 3: 69.0% hemolytic) that this reward term is meant to counteract.

Model/tokenizer loaded once at import (mirrors model2_client.py pattern).
Each score_batch() call uses a fresh temp working directory so concurrent
or repeated calls never collide on intermediate files.
"""

import os
import sys
import tempfile
import subprocess
import pandas as pd
import numpy as np

HEMOPI2_SCRIPTS = os.path.expanduser(
    "~/miniconda3/envs/toxin_hemo/lib/python3.10/site-packages/hemopi2/python_scripts"
)
HEMOPI2_ROOT = os.path.dirname(HEMOPI2_SCRIPTS)

if not os.path.isdir(HEMOPI2_SCRIPTS):
    raise RuntimeError(
        f"hemopi2 package scripts not found at {HEMOPI2_SCRIPTS} - "
        "update HEMOPI2_SCRIPTS to match your conda env path."
    )

sys.path.insert(0, HEMOPI2_SCRIPTS)

import hemopi2_classification as h2  # noqa: E402

MERCI_BIN = os.path.join(HEMOPI2_ROOT, "merci", "MERCI_motif_locator.pl")
MOTIFS_P1 = os.path.join(HEMOPI2_ROOT, "motif", "pos_motif_1.txt")
MOTIFS_N1 = os.path.join(HEMOPI2_ROOT, "motif", "neg_motif_1.txt")
MOTIFS_P2 = os.path.join(HEMOPI2_ROOT, "motif", "pos_motif_2.txt")
MOTIFS_N2 = os.path.join(HEMOPI2_ROOT, "motif", "neg_motif_2.txt")

DEFAULT_THRESHOLD = 0.55


def _run_perl(cmd: str):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MERCI call failed: {cmd}\nstderr: {result.stderr}")


def score_batch(sequences: list, threshold: float = DEFAULT_THRESHOLD) -> list:
    """Returns Hybrid2 (ESM+MERCI) hemolysis probability per sequence,
    in the same order as `sequences`, matching
    `hemopi2_classification -m 4 -t <threshold>` output."""
    if not sequences:
        return []

    with tempfile.TemporaryDirectory() as wd:
        seqids = [f"seq_{i}" for i in range(len(sequences))]
        fasta_path = os.path.join(wd, "input.fasta")
        with open(fasta_path, "w") as f:
            for sid, seq in zip(seqids, sequences):
                f.write(f">{sid}\n{seq}\n")

        df_2, dfseq = h2.readseq(fasta_path)
        df1 = h2.lenchk(dfseq, wd)

        # main() in the original CLI writes Sequence_1 itself, outside any
        # named helper function - replicate that step here, since it's a
        # required input file for the MERCI subprocess calls below.
        seqid_1 = list(map(">{}".format, seqids))
        CM = pd.concat([pd.DataFrame(seqid_1), pd.DataFrame(sequences)], axis=1)
        CM.to_csv(os.path.join(wd, "Sequence_1"), header=False, index=None, sep="\n")

        esm_out_path = os.path.join(wd, "esm_out.csv")
        h2.run_esm_model(dfseq, df_2, esm_out_path, threshold)
        df3 = pd.read_csv(esm_out_path)

        _run_perl(f"perl {MERCI_BIN} -p {wd}/Sequence_1 -i {MOTIFS_P1} -o {wd}/merci_p_1.txt")
        _run_perl(f"perl {MERCI_BIN} -p {wd}/Sequence_1 -i {MOTIFS_N1} -o {wd}/merci_n_1.txt")
        _run_perl(f"perl {MERCI_BIN} -p {wd}/Sequence_1 -i {MOTIFS_P2} -c KOOLMAN-ROHM -o {wd}/merci_p_2.txt")
        _run_perl(f"perl {MERCI_BIN} -p {wd}/Sequence_1 -i {MOTIFS_N2} -c KOOLMAN-ROHM -o {wd}/merci_n_2.txt")

        h2.MERCI_Processor_p(wd, "merci_p_1.txt", "/merci_output_p_1.csv", seqids)
        h2.MERCI_Processor_p(wd, "merci_p_2.txt", "/merci_output_p_2.csv", seqids)
        h2.Merci_after_processing_p(wd, "/merci_output_p_1.csv", "/merci_hybrid_p_1.csv")
        h2.Merci_after_processing_p(wd, "/merci_output_p_2.csv", "/merci_hybrid_p_2.csv")

        h2.MERCI_Processor_n(wd, "/merci_n_1.txt", "/merci_output_n_1.csv", seqids)
        h2.MERCI_Processor_n(wd, "/merci_n_2.txt", "/merci_output_n_2.csv", seqids)
        h2.Merci_after_processing_n(wd, "/merci_output_n_1.csv", "/merci_hybrid_n_1.csv")
        h2.Merci_after_processing_n(wd, "/merci_output_n_2.csv", "/merci_hybrid_n_2.csv")

        h2.hybrid(
            wd, df3,
            "/merci_hybrid_p_1.csv", "/merci_hybrid_n_1.csv",
            "/merci_hybrid_p_2.csv", "/merci_hybrid_n_2.csv",
            threshold, "/final_output",
        )

        df_final = pd.read_csv(os.path.join(wd, "final_output"))
        df_final.loc[df_final["Hybrid Score"] > 1, "Hybrid Score"] = 1
        df_final.loc[df_final["Hybrid Score"] < 0, "Hybrid Score"] = 0

        df_final["SeqID"] = df_final["SeqID"].astype(str).str.replace(">", "", regex=False)
        score_by_id = dict(zip(df_final["SeqID"], df_final["Hybrid Score"]))
        scores = [float(score_by_id.get(sid, 0.0)) for sid in seqids]

        return scores


if __name__ == "__main__":
    test_seqs = [
        "GIGKKILRIGKILKNLFKGIGK",
        "ACDEFGHIKLMNPQRSTVWY",
    ]
    scores = score_batch(test_seqs)
    for seq, score in zip(test_seqs, scores):
        print(f"{score:.3f}  {seq}")
