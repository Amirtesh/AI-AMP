#!/usr/bin/env python3
"""
cluster_amp_families.py
Unsupervised clustering of the AMP-positive training corpus in ESM2
embedding space, as a data-driven proxy for structural/compositional AMP
families - used to build a class-diversity reward term for a retrained
GRPO run. Does not touch or replace any existing script.

Usage:
    python3 cluster_amp_families.py --data amp_positive_filtered.csv \
        --k_range 5 8 10 15 20 --output_prefix amp_clusters
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import esm2_embedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="amp_positive_filtered.csv")
    ap.add_argument("--seq_column", default="sequence")
    ap.add_argument("--k_range", type=int, nargs="+", default=[5, 8, 10, 15, 20],
                     help="Candidate cluster counts to try - silhouette score "
                          "picks the best, but inspect the printed cluster "
                          "samples yourself before trusting the number blindly.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_prefix", default="amp_clusters")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df[args.seq_column] = df[args.seq_column].astype(str).str.strip().str.upper()
    sequences = df[args.seq_column].tolist()
    print(f"Embedding {len(sequences)} sequences via ESM2 (this is the same "
          f"pipeline Model2/AD-penalty already use)...")

    embeddings = esm2_embedder.get_embeddings(sequences)
    np.save(f"{args.output_prefix}_embeddings.npy", embeddings)
    print(f"Saved raw embeddings to {args.output_prefix}_embeddings.npy "
          f"(reusable if you want to try different K later without re-embedding)")

    results = []
    best_k, best_score, best_labels, best_model = None, -1, None, None

    for k in args.k_range:
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=min(5000, len(embeddings)),
                                  random_state=args.seed)
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        print(f"k={k}: silhouette={score:.4f}, cluster sizes={cluster_sizes.tolist()}")
        results.append({"k": k, "silhouette": score})

        if score > best_score:
            best_k, best_score, best_labels, best_model = k, score, labels, km

    print(f"\nBest k by silhouette: {best_k} (score={best_score:.4f})")
    print("NOTE: silhouette score alone doesn't guarantee biological meaningfulness - "
          "inspect the printed sample sequences per cluster below before committing.")

    df["cluster"] = best_labels
    df[[args.seq_column, "cluster"]].to_csv(f"{args.output_prefix}_assignments.csv", index=False)
    pd.DataFrame(results).to_csv(f"{args.output_prefix}_k_selection.csv", index=False)

    import joblib
    joblib.dump(best_model, f"{args.output_prefix}_kmeans_k{best_k}.joblib")
    print(f"\nSaved: {args.output_prefix}_assignments.csv, "
          f"{args.output_prefix}_k_selection.csv, "
          f"{args.output_prefix}_kmeans_k{best_k}.joblib")

    # --- Print sample sequences per cluster for manual inspection ---
    print(f"\n=== Sample sequences per cluster (k={best_k}) ===")
    for c in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == c]
        samples = cluster_df[args.seq_column].sample(min(5, len(cluster_df)), random_state=args.seed).tolist()
        print(f"\nCluster {c} (n={len(cluster_df)}, "
              f"mean_len={cluster_df[args.seq_column].str.len().mean():.1f}):")
        for s in samples:
            print(f"  {s}")


if __name__ == "__main__":
    main()
