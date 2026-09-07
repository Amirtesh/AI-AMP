#!/usr/bin/env python3
"""
test_reward_components.py
Standalone sanity checks for peptide_features.py and diversity_penalty.py -
no server, no ESM2, no GPU needed. Run this before touching reward.py or
grpo_train.py.
"""

from peptide_features import (
    net_charge, hydrophobic_ratio, hydrophobic_moment,
    length_score, charge_score, compute_physicochemical_terms,
)
from diversity_penalty import hamming_similarity, diversity_penalty_batch


def check(label, actual, expected, tol=1e-6):
    ok = abs(actual - expected) < tol if isinstance(expected, float) else actual == expected
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: got {actual}, expected {expected}")
    return ok


def main():
    all_pass = True

    print("=== net_charge ===")
    all_pass &= check("all-K (5x)", net_charge("KKKKK"), 5.0)
    all_pass &= check("all-D (5x)", net_charge("DDDDD"), -5.0)
    all_pass &= check("K+D cancel", net_charge("KD"), 0.0)
    all_pass &= check("neutral (no charged residues)", net_charge("GAVLI"), 0.0)
    all_pass &= check("H partial charge", net_charge("H"), 0.1)
    print()

    print("=== hydrophobic_ratio ===")
    all_pass &= check("all hydrophobic", hydrophobic_ratio("AILMFWVCY"), 1.0)
    all_pass &= check("all polar/charged", hydrophobic_ratio("KRDESTQN"), 0.0)
    all_pass &= check("half-half", hydrophobic_ratio("AK"), 0.5)
    all_pass &= check("empty string", hydrophobic_ratio(""), 0.0)
    print()

    print("=== hydrophobic_moment (sanity, not exact-value check) ===")
    # A sequence alternating hydrophobic/hydrophilic every ~3.6 residues
    # (alpha-helix period) should show HIGHER moment than a sequence with
    # the same composition but poorly-clustered hydrophobic residues.
    amphipathic = "LKKLLKLLKKLLKL"   # hydrophobic/charged alternating, helix-like
    scrambled = "LLLLKKKKKKLLLL"      # same composition, blocked not alternating
    mom_amphi = hydrophobic_moment(amphipathic)
    mom_scrambled = hydrophobic_moment(scrambled)
    print(f"  amphipathic-pattern moment: {mom_amphi:.4f}")
    print(f"  blocked-pattern moment:     {mom_scrambled:.4f}")
    print(f"  [{'PASS' if mom_amphi > mom_scrambled else 'CHECK MANUALLY'}] "
          f"amphipathic arrangement should generally score higher "
          f"(not a strict guarantee for every sequence pair, but expected here)")
    print()

    print("=== length_score ===")
    all_pass &= check("exactly at min (15)", length_score("A" * 15), 1.0)
    all_pass &= check("exactly at max (50)", length_score("A" * 50), 1.0)
    all_pass &= check("mid-range (30)", length_score("A" * 30), 1.0)
    all_pass &= check("2 below min, margin 5", length_score("A" * 13), 0.6)
    all_pass &= check("5 below min = 0", length_score("A" * 10), 0.0)
    all_pass &= check("10 below min, clipped to 0", length_score("A" * 5), 0.0)
    all_pass &= check("2 above max, margin 5", length_score("A" * 52), 0.6)
    print()

    print("=== charge_score ===")
    all_pass &= check("at min charge (+2)", charge_score("KD" + "A" * 10), 1.0)  # net +... check below
    # simpler direct constructions:
    all_pass &= check("net charge +5 (mid-range)", charge_score("K" * 5), 1.0)
    all_pass &= check("net charge 0 (2 below min=2, margin=2)", charge_score(""), 0.0)
    all_pass &= check("net charge +9 (at max)", charge_score("K" * 9), 1.0)
    all_pass &= check("net charge +11 (2 above max, margin 2)", charge_score("K" * 11), 0.0)
    print()

    print("=== compute_physicochemical_terms (integration check) ===")
    test_seq = "GIGKFLHSAKKFGKAFVGEIMNS"
    terms = compute_physicochemical_terms(test_seq)
    print(f"  Sequence: {test_seq}")
    for k, v in terms.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print()

    print("=== hamming_similarity ===")
    all_pass &= check("identical sequences", hamming_similarity("KKKAAA", "KKKAAA"), 1.0)
    all_pass &= check("completely different, same length",
                       hamming_similarity("AAAAAA", "KKKKKK"), 0.0)
    all_pass &= check("half matching", hamming_similarity("AAAKKK", "AAAAAA"), 0.5)
    print()

    print("=== diversity_penalty_batch ===")
    group_with_duplicates = [
        "KKLLKKLLKKLLK",
        "KKLLKKLLKKLLK",  # exact duplicate of above
        "AGDESTQNPVCWY",  # very different
    ]
    penalties = diversity_penalty_batch(group_with_duplicates, similarity_threshold=0.7)
    print(f"  Sequences: {group_with_duplicates}")
    print(f"  Penalties: {[f'{p:.3f}' for p in penalties]}")
    dup_ok = penalties[0] > 0 and penalties[1] > 0 and penalties[2] == 0.0
    print(f"  [{'PASS' if dup_ok else 'FAIL'}] near-identical pair penalized, "
          f"distinct sequence unpenalized")
    all_pass &= dup_ok
    print()

    print("=" * 50)
    print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED - review above before proceeding")


if __name__ == "__main__":
    main()
