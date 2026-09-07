#!/usr/bin/env python3
"""
compare_checkpoints.py
Generates samples from multiple GRPO checkpoints and compares them on
diversity + reward metrics, not just mean reward - specifically built to
catch compositional mode collapse (narrow AA vocabulary, length clustering)
that the training-time diversity_penalty doesn't detect, since it only
checks within-group literal similarity, not compositional diversity across
a whole generation batch.

Usage:
    python3 compare_checkpoints.py --checkpoint_dir grpo_checkpoints \
        --steps 0 25 50 100 150 200 300 400 499 --n_samples 200
"""

import argparse
import math
import os
from collections import Counter
from itertools import combinations

import pandas as pd
import torch
import torch.nn.functional as F

from train_stage1 import SmallPeptideGPT, BOS_IDX, EOS_IDX, VOCAB_SIZE, IDX2TOKEN, AMINO_ACIDS
from reward import compute_rewards
import class_diversity
from diversity_penalty import hamming_similarity

STAGE1_MAX_LEN = 83


def load_policy(checkpoint_path, device):
    model = SmallPeptideGPT(VOCAB_SIZE, STAGE1_MAX_LEN + 2, d_model=256, n_heads=8, n_layers=6)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["policy_state"] if "policy_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_one(model, device, min_len, max_len, temperature=1.0, top_p=0.9):
    ids = [BOS_IDX]
    for _ in range(max_len):
        x = torch.tensor([ids], device=device)
        logits = model(x)[0, -1] / temperature
        if len(ids) - 1 < min_len:
            logits = logits.clone()
            logits[EOS_IDX] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = (cumsum > top_p).nonzero()[0].item() + 1
        sorted_probs[cutoff:] = 0
        sorted_probs /= sorted_probs.sum()
        next_id = sorted_idx[torch.multinomial(sorted_probs, 1)].item()
        if next_id == EOS_IDX:
            break
        ids.append(next_id)
    return "".join(IDX2TOKEN[i] for i in ids[1:])


def aa_composition_entropy(sequences):
    """Shannon entropy of pooled amino-acid frequency across ALL generated
    sequences - low entropy means the model is leaning on a narrow subset
    of the 20 amino acids, regardless of how many distinct exact strings
    it produces."""
    counts = Counter()
    for seq in sequences:
        counts.update(seq)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
    max_entropy = math.log2(len(AMINO_ACIDS))  # entropy if all 20 AAs equally used
    return entropy / max_entropy  # normalized to [0, 1]


def dipeptide_diversity(sequences):
    """Fraction of unique dipeptides (2-mers) out of all dipeptides observed -
    a second, independent view of motif-level (not just single-residue)
    compositional diversity."""
    all_dipeptides = []
    for seq in sequences:
        all_dipeptides.extend(seq[i:i+2] for i in range(len(seq) - 1))
    if not all_dipeptides:
        return 0.0
    return len(set(all_dipeptides)) / len(all_dipeptides)


def mean_pairwise_similarity(sequences, max_pairs=2000, seed=42):
    """Average Hamming-style similarity over a random sample of pairs -
    full O(n^2) is wasteful at n_samples=200+, so subsample pairs."""
    import random
    rng = random.Random(seed)
    n = len(sequences)
    all_pairs = list(combinations(range(n), 2))
    if len(all_pairs) > max_pairs:
        all_pairs = rng.sample(all_pairs, max_pairs)
    if not all_pairs:
        return 0.0
    sims = [hamming_similarity(sequences[i], sequences[j]) for i, j in all_pairs]
    return sum(sims) / len(sims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default="grpo_checkpoints")
    ap.add_argument("--steps", type=int, nargs="+", required=True,
                     help="Which step_N.pt checkpoints to compare, e.g. 0 25 100 499")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--min_len", type=int, default=15)
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--output_csv", default="checkpoint_comparison.csv")
    ap.add_argument("--save_sequences_dir", default="checkpoint_samples")
    ap.add_argument("--cluster_kmeans", default="amp_clusters_kmeans_k5.joblib")
    ap.add_argument("--cluster_assignments", default="amp_clusters_assignments.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_sequences_dir, exist_ok=True)
    class_diversity.load_cluster_model(args.cluster_kmeans, args.cluster_assignments)

    rows = []

    for step in args.steps:
        ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[skip] {ckpt_path} not found")
            continue

        print(f"\n=== step_{step}.pt ===")
        model = load_policy(ckpt_path, device)

        sequences = [
            generate_one(model, device, args.min_len, args.max_len, args.temperature)
            for _ in range(args.n_samples)
        ]
        sequences = [s for s in sequences if s]  # drop any empty edge cases

        pd.DataFrame({"sequence": sequences}).to_csv(
            os.path.join(args.save_sequences_dir, f"step_{step}_samples.csv"), index=False
        )

        lengths = [len(s) for s in sequences]
        unique_ratio = len(set(sequences)) / len(sequences) if sequences else 0.0
        aa_entropy = aa_composition_entropy(sequences)
        dipep_div = dipeptide_diversity(sequences)
        pairwise_sim = mean_pairwise_similarity(sequences)

        rewards, breakdown = compute_rewards(sequences)

        row = {
            "step": step,
            "n_samples": len(sequences),
            "unique_seq_ratio": unique_ratio,
            "mean_length": sum(lengths) / len(lengths) if lengths else 0,
            "std_length": pd.Series(lengths).std() if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "aa_composition_entropy_normalized": aa_entropy,  # 1.0 = all 20 AAs equally used
            "dipeptide_diversity": dipep_div,                  # higher = more distinct motifs
            "mean_pairwise_similarity": pairwise_sim,           # lower = more diverse
            "mean_reward": sum(rewards) / len(rewards),
            "mean_model2": sum(breakdown["model2"]) / len(breakdown["model2"]),
            "mean_charge": sum(breakdown["charge"]) / len(breakdown["charge"]),
            "mean_length_score": sum(breakdown["length"]) / len(breakdown["length"]),
            "mean_hmom": sum(breakdown["hydrophobic_moment"]) / len(breakdown["hydrophobic_moment"]),
        }
        rows.append(row)

        print(f"  reward={row['mean_reward']:.3f}  "
              f"aa_entropy={aa_entropy:.3f}  "
              f"dipeptide_div={dipep_div:.3f}  "
              f"pairwise_sim={pairwise_sim:.3f}  "
              f"len={row['mean_length']:.1f}±{row['std_length']:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved comparison to {args.output_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
