#!/usr/bin/env python3
"""
ad_penalty.py
Applicability-domain penalty: penalizes generated sequences that sit close,
in ESM2 embedding space, to Model 2's confirmed-inactive Tier 1 negatives -
directly operationalizing the Section 10.6 finding that Model 2 is
confidently wrong on cationic peptides resembling known failed candidates.

*** INTEGRATION POINT ***
get_esm2_embeddings() below is a stub - point it at your existing ESM2
embedding extraction code (the same pipeline used for Model 2's
gram_negative feature set). Must return an (N, 480) array for
esm2_t12_35M_UR50D, one row per input sequence, in input order.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

_nn_index = None  # lazy-built on first use, cached for the training run


def get_esm2_embeddings(sequences):
    """*** REPLACE WITH YOUR ACTUAL ESM2 EMBEDDING PIPELINE ***
    Placeholder raises explicitly rather than returning fake data, so a
    forgotten integration point fails loudly instead of silently
    corrupting the AD-penalty term."""
    raise NotImplementedError(
        "Wire this up to your existing ESM2 embedding extraction code "
        "(esm2_t12_35M_UR50D, same as Model 2's gram_negative features)."
    )


def build_negative_index(tier1_negative_csv, seq_column="sequence"):
    """Precompute embeddings for Model 2's 585 Tier 1 confirmed negatives
    and build a nearest-neighbor index. Call this once at the start of
    training, not per-step."""
    global _nn_index
    import pandas as pd
    df = pd.read_csv(tier1_negative_csv)
    seqs = df[seq_column].astype(str).str.strip().str.upper().tolist()
    embeddings = get_esm2_embeddings(seqs)
    _nn_index = NearestNeighbors(n_neighbors=1, metric="cosine")
    _nn_index.fit(embeddings)
    print(f"AD-penalty index built from {len(seqs)} confirmed-negative sequences")


def ad_penalty_batch(sequences, danger_threshold=0.15):
    """Returns a penalty in [0, 1] per sequence - 1.0 if very close to a
    known confirmed-inactive peptide (cosine distance below threshold),
    scaling down to 0.0 as distance grows. Subtract this from reward,
    don't multiply - a near-miss shouldn't zero out an otherwise-good
    physicochemical profile, just discourage it."""
    if _nn_index is None:
        raise RuntimeError("Call build_negative_index() before scoring - "
                            "index not yet built.")
    embeddings = get_esm2_embeddings(sequences)
    distances, _ = _nn_index.kneighbors(embeddings)
    distances = distances.flatten()
    penalties = np.clip(1.0 - distances / danger_threshold, 0.0, 1.0)
    return penalties.tolist()
