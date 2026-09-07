#!/usr/bin/env python3
"""
class_diversity.py
Class-diversity reward term - pushes GRPO to keep producing sequences
across the full set of empirically-derived AMP clusters (from
cluster_amp_families.py), rather than converging onto one dominant family.
Same architectural pattern as ad_penalty.py: build a reference index once,
score against it per step.

Two components:
1. Underrepresentation bonus - reward generating into a cluster that's
   rare in the natural training corpus (targets cluster 2/defensin-like,
   currently near-absent from GRPO output).
2. Running visitation bonus - reward generating into a cluster this
   training RUN has visited less, so far - directly targets the observed
   failure mode (global drift toward one mode over training), which the
   existing group-local diversity_penalty structurally cannot catch.
"""

import numpy as np
import joblib

import esm2_embedder

_kmeans = None
_cluster_natural_freq = None   # natural frequency of each cluster in training corpus
_visitation_counts = None       # running count, updated during THIS training run


def load_cluster_model(kmeans_path, assignments_csv):
    """Call once at the start of a GRPO run."""
    global _kmeans, _cluster_natural_freq, _visitation_counts
    import pandas as pd

    _kmeans = joblib.load(kmeans_path)
    k = _kmeans.n_clusters

    assignments = pd.read_csv(assignments_csv)
    counts = assignments["cluster"].value_counts().reindex(range(k), fill_value=0)
    _cluster_natural_freq = (counts / counts.sum()).values  # shape (k,)

    _visitation_counts = np.ones(k)  # start at 1, not 0 - avoids div-by-zero, mild Laplace smoothing
    print(f"Class-diversity index loaded: k={k}, natural frequencies={_cluster_natural_freq.round(3)}")


def assign_clusters(sequences):
    """Returns cluster index per sequence. Empty/degenerate sequences are
    assigned cluster -1 (not a real cluster) rather than crashing on the
    resulting NaN embedding."""
    if _kmeans is None:
        raise RuntimeError("Call load_cluster_model() before scoring.")
    valid_idx = [i for i, s in enumerate(sequences) if len(s) > 0]
    clusters = np.full(len(sequences), -1, dtype=int)
    if valid_idx:
        valid_seqs = [sequences[i] for i in valid_idx]
        embeddings = esm2_embedder.get_embeddings(valid_seqs)
        valid_clusters = _kmeans.predict(embeddings)
        for idx, c in zip(valid_idx, valid_clusters):
            clusters[idx] = c
    return clusters


def class_diversity_batch(sequences, underrep_weight=0.5, visitation_weight=0.5):
    """Returns a bonus in roughly [0, 1] per sequence, combining:
    - underrepresentation bonus: 1 - natural_frequency of its cluster
      (higher reward for rare natural classes, e.g. defensin-like)
    - visitation bonus: inverse of how often THIS RUN has hit that
      cluster so far (higher reward for clusters this run has neglected)
    Also updates the running visitation count - call this once per GRPO
    step, in order, not out of sequence, since it's stateful.
    """
    global _visitation_counts
    if _kmeans is None:
        raise RuntimeError("Call load_cluster_model() before scoring.")

    clusters = assign_clusters(sequences)
    bonuses = []

    for c in clusters:
        underrep_bonus = 1.0 - _cluster_natural_freq[c]
        visitation_bonus = 1.0 / np.sqrt(_visitation_counts[c])  # sqrt for gentler decay
        bonus = underrep_weight * underrep_bonus + visitation_weight * visitation_bonus
        bonuses.append(bonus)
        _visitation_counts[c] += 1  # update AFTER computing this sequence's bonus

    # Normalize roughly to [0, 1] range for comparability with other reward terms
    bonuses = np.array(bonuses)
    max_possible = underrep_weight * 1.0 + visitation_weight * 1.0  # visitation starts at 1.0 max
    return (bonuses / max_possible).tolist(), clusters.tolist()
