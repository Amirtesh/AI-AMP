#!/usr/bin/env python3
"""
diversity_penalty.py
Within-group diversity term - penalizes sequences that are near-duplicates
of others generated in the same GRPO group, discouraging mode collapse
independent of the main reward signal.
"""

def _dipeptide_vector(seq):
    """Dipeptide composition vector - captures motif/compositional
    similarity regardless of positional shift or indel, unlike Hamming
    which is blind to shifted-but-identical motifs and was found to
    never fire during GRPO v1 despite confirmed compositional collapse."""
    from collections import Counter
    if len(seq) < 2:
        return Counter()
    return Counter(seq[i:i+2] for i in range(len(seq) - 1))

def hamming_similarity(seq_a, seq_b):
    """Cosine similarity over dipeptide composition vectors - replaces
    the old Hamming-based proxy. Name kept for compatibility with
    diversity_penalty_batch below."""
    import math
    if not seq_a or not seq_b:
        return 0.0
    va, vb = _dipeptide_vector(seq_a), _dipeptide_vector(seq_b)
    keys = set(va) | set(vb)
    if not keys:
        return 0.0
    dot = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def diversity_penalty_batch(sequences, similarity_threshold=0.7):
    """Returns a penalty per sequence based on its highest similarity to
    any OTHER sequence in the same batch/group."""
    n = len(sequences)
    penalties = [0.0] * n
    for i in range(n):
        max_sim = 0.0
        for j in range(n):
            if i == j:
                continue
            sim = hamming_similarity(sequences[i], sequences[j])
            max_sim = max(max_sim, sim)
        if max_sim > similarity_threshold:
            penalties[i] = (max_sim - similarity_threshold) / (1.0 - similarity_threshold)
    return penalties
