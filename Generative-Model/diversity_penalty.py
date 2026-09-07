#!/usr/bin/env python3
"""
diversity_penalty.py
Within-group diversity term - penalizes sequences that are near-duplicates
of others generated in the same GRPO group, discouraging mode collapse
independent of the main reward signal.
"""

def hamming_similarity(seq_a, seq_b):
    """Simple, fast proxy for similarity between equal-or-near-length
    sequences - not a substitute for CD-HIT, just cheap enough to run
    O(group_size^2) times per GRPO step without becoming the bottleneck."""
    if not seq_a or not seq_b:
        return 0.0
    shorter, longer = sorted([seq_a, seq_b], key=len)
    matches = sum(1 for a, b in zip(shorter, longer) if a == b)
    return matches / len(longer)


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
