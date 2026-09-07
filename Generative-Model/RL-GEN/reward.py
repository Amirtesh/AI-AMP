#!/usr/bin/env python3
"""
reward.py
Composite reward function - Model2 gatekeeper score, physicochemical
shaping terms (charge/length/hydrophobic moment), class-diversity term,
and CLI-overridable weights. AD-penalty remains disabled per earlier
finding (see reward_legacy_v1.py for the original AD-penalty test).
"""

from peptide_features import compute_physicochemical_terms
from model2_client import score_batch
import diversity_penalty
import class_diversity
import hemopi2_client
import toxinpred3_client

WEIGHTS = {
    "model2": 1.0,
    "charge": 0.3,
    "length": 0.2,
    "hydrophobic_moment": 0.2,
    "diversity_penalty": 0.3,
    "class_diversity": 0.25,   # reintroduced after confirming bug fixes alone slow but do not stop collapse (cluster 4 concentration 79.3%->91.4% over steps 75-299)
    "hemolysis_penalty": 0.4,   # NEW - directly penalizes predicted hemolysis (HemoPI2 Hybrid2)
    "toxicity_penalty": 0.4,   # NEW - directly penalizes predicted toxicity (ToxinPred3 Hybrid)
}


def compute_rewards(sequences, weights=None):
    """weights: optional dict overriding module-level WEIGHTS defaults.
    Keys not present in the override fall back to WEIGHTS[key]."""
    w = {**WEIGHTS, **(weights or {})}
    n = len(sequences)
    model2_scores = score_batch(sequences)
    phys_terms = [compute_physicochemical_terms(s) for s in sequences]
    div_penalties = diversity_penalty.diversity_penalty_batch(sequences)
    class_div_bonuses, cluster_ids = class_diversity.class_diversity_batch(sequences)
    hemolysis_scores = hemopi2_client.score_batch(sequences)
    toxicity_scores = toxinpred3_client.score_batch(sequences)

    rewards = []
    breakdown = {k: [] for k in list(WEIGHTS.keys()) + ["total", "cluster_id"]}

    for i in range(n):
        r_model2 = model2_scores[i]
        r_charge = phys_terms[i]["charge_score"]
        r_length = phys_terms[i]["length_score"]
        r_hmom = min(phys_terms[i]["hydrophobic_moment"] / 0.65, 1.0)  # updated to real-AMP 90th percentile, see reward.py
        r_div = -div_penalties[i]
        r_class = class_div_bonuses[i]
        r_hemolysis = 1.0 - hemolysis_scores[i]
        r_toxicity = 1.0 - toxicity_scores[i]

        total = (
            w["model2"] * r_model2
            + w["charge"] * r_charge
            + w["length"] * r_length
            + w["hydrophobic_moment"] * r_hmom
            + w["diversity_penalty"] * r_div
            + w["class_diversity"] * r_class
            + w["hemolysis_penalty"] * r_hemolysis
            + w["toxicity_penalty"] * r_toxicity
        )

        breakdown["model2"].append(r_model2)
        breakdown["charge"].append(r_charge)
        breakdown["length"].append(r_length)
        breakdown["hydrophobic_moment"].append(r_hmom)
        breakdown["diversity_penalty"].append(r_div)
        breakdown["class_diversity"].append(r_class)
        breakdown["hemolysis_penalty"].append(r_hemolysis)
        breakdown["toxicity_penalty"].append(r_toxicity)
        breakdown["cluster_id"].append(cluster_ids[i])
        breakdown["total"].append(total)
        rewards.append(total)

    return rewards, breakdown
