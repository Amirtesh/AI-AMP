#!/usr/bin/env python3
"""
reward.py
Composite reward function - combines Model 2 score, physicochemical
shaping terms, AD-penalty, and diversity penalty into one scalar reward
per sequence. This is what grpo_train.py calls once per group, per step.
"""

from peptide_features import compute_physicochemical_terms
from model2_client import score_batch
import ad_penalty
import diversity_penalty

# Weights - starting values, NOT tuned. Log every term separately during
# early GRPO runs and adjust based on what's actually driving reward,
# rather than trusting these numbers blindly.
WEIGHTS = {
    "model2": 1.0,
    "charge": 0.3,
    "length": 0.2,
    "hydrophobic_moment": 0.2,
    "ad_penalty": 0.0,  # disabled - embedding-distance AD-penalty showed no discriminative power (real vs scrambled AMPs indistinguishable); see project notes
    "diversity_penalty": 0.3,
}


def compute_rewards(sequences, use_ad_penalty=False):  # disabled by default - see reward.py note
    """Returns (rewards: list[float], term_breakdown: dict[str, list[float]])
    - breakdown is returned for logging/debugging, not just the final scalar,
    since silently trusting one number without seeing its components is
    exactly the kind of thing that hides a broken reward term."""
    n = len(sequences)
    model2_scores = score_batch(sequences)
    phys_terms = [compute_physicochemical_terms(s) for s in sequences]

    if use_ad_penalty:
        ad_penalties = ad_penalty.ad_penalty_batch(sequences)
    else:
        ad_penalties = [0.0] * n

    div_penalties = diversity_penalty.diversity_penalty_batch(sequences)

    rewards = []
    breakdown = {k: [] for k in list(WEIGHTS.keys()) + ["total"]}

    for i in range(n):
        r_model2 = model2_scores[i]
        r_charge = phys_terms[i]["charge_score"]
        r_length = phys_terms[i]["length_score"]
        r_hmom = min(phys_terms[i]["hydrophobic_moment"] / 0.65, 1.0)  # normalize using real-AMP 90th percentile (0.641), was 0.5 which sat below the 75th pct and caused early saturation
        r_ad = -ad_penalties[i]
        r_div = -div_penalties[i]

        total = (
            WEIGHTS["model2"] * r_model2
            + WEIGHTS["charge"] * r_charge
            + WEIGHTS["length"] * r_length
            + WEIGHTS["hydrophobic_moment"] * r_hmom
            + WEIGHTS["ad_penalty"] * r_ad
            + WEIGHTS["diversity_penalty"] * r_div
        )

        breakdown["model2"].append(r_model2)
        breakdown["charge"].append(r_charge)
        breakdown["length"].append(r_length)
        breakdown["hydrophobic_moment"].append(r_hmom)
        breakdown["ad_penalty"].append(r_ad)
        breakdown["diversity_penalty"].append(r_div)
        breakdown["total"].append(total)
        rewards.append(total)

    return rewards, breakdown
