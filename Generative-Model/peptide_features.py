#!/usr/bin/env python3
"""
peptide_features.py
Physicochemical feature computation for reward shaping - reusable,
standalone, testable independent of the RL loop.
"""

CHARGE_POSITIVE = {"K": 1.0, "R": 1.0, "H": 0.1}   # H partial, per standard pKa treatment
CHARGE_NEGATIVE = {"D": -1.0, "E": -1.0}
HYDROPHOBIC_SET = set("AILMFWVCY")  # standard hydrophobic residues

# Eisenberg consensus hydrophobicity scale, for hydrophobic moment calculation
EISENBERG_SCALE = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}


def net_charge(seq):
    return sum(CHARGE_POSITIVE.get(c, 0) for c in seq) + sum(CHARGE_NEGATIVE.get(c, 0) for c in seq)


def hydrophobic_ratio(seq):
    if not seq:
        return 0.0
    return sum(1 for c in seq if c in HYDROPHOBIC_SET) / len(seq)


def hydrophobic_moment(seq, angle_deg=100.0):
    """Eisenberg hydrophobic moment - standard measure of amphipathicity,
    assuming an alpha-helical structure (100 deg/residue turn)."""
    import math
    if len(seq) < 2:
        return 0.0
    angle_rad = math.radians(angle_deg)
    sum_cos, sum_sin = 0.0, 0.0
    for i, c in enumerate(seq):
        h = EISENBERG_SCALE.get(c, 0.0)
        sum_cos += h * math.cos(i * angle_rad)
        sum_sin += h * math.sin(i * angle_rad)
    return math.sqrt(sum_cos**2 + sum_sin**2) / len(seq)


def length_score(seq, target_min=15, target_max=50, soft_margin=5):
    """Soft length shaping - full credit inside [target_min, target_max],
    linear falloff over soft_margin residues outside it. Generation is
    already hard-constrained via min/max in sampling, so this mainly
    matters if that constraint is ever relaxed during exploration."""
    L = len(seq)
    if target_min <= L <= target_max:
        return 1.0
    if L < target_min:
        return max(0.0, 1.0 - (target_min - L) / soft_margin)
    return max(0.0, 1.0 - (L - target_max) / soft_margin)


def charge_score(seq, target_min=2.0, target_max=9.0, soft_margin=2.0):
    """Soft charge shaping, same falloff logic as length_score."""
    c = net_charge(seq)
    if target_min <= c <= target_max:
        return 1.0
    if c < target_min:
        return max(0.0, 1.0 - (target_min - c) / soft_margin)
    return max(0.0, 1.0 - (c - target_max) / soft_margin)


def compute_physicochemical_terms(seq):
    """Returns a dict of individual terms - kept separate (not pre-summed)
    so the reward function can weight/combine them explicitly and log each
    term independently for debugging."""
    return {
        "charge": net_charge(seq),
        "charge_score": charge_score(seq),
        "hydrophobic_ratio": hydrophobic_ratio(seq),
        "hydrophobic_moment": hydrophobic_moment(seq),
        "length_score": length_score(seq),
    }
