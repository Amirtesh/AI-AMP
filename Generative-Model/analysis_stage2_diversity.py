import sys
sys.path.insert(0, "model2")
sys.path.insert(0, "model3_analysis")

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

from esm2_embedder import get_embeddings

km = joblib.load("model3_analysis/amp_clusters_kmeans_k5.joblib")

samples = {
    "Sample 1": "gen_sample_1.csv",
    "Sample 2": "gen_sample_2.csv",
}

results = {}
all_embeddings = []
all_labels = []
all_sources = []

for name, path in samples.items():
    df = pd.read_csv(path)
    seq_col = "sequence" if "sequence" in df.columns else df.columns[0]
    seqs = df[seq_col].dropna().unique().tolist()
    embs = get_embeddings(seqs)
    clusters = km.predict(embs)

    vals, counts = np.unique(clusters, return_counts=True)
    pct = {v: 0.0 for v in range(5)}
    for v, c in zip(vals, counts):
        pct[v] = 100 * c / len(clusters)
    results[name] = pct
    print(f"{name} ({len(seqs)} seqs): {pct}")

    all_embeddings.append(embs)
    all_labels.append(clusters)
    all_sources.extend([name] * len(seqs))

# --- Bar plot: cluster % per sample, side by side ---
fig, ax = plt.subplots(figsize=(8, 5))
clusters_idx = list(range(5))
width = 0.35
x = np.arange(5)

for i, (name, pct) in enumerate(results.items()):
    vals = [pct[c] for c in clusters_idx]
    ax.bar(x + i * width, vals, width, label=name)

ax.set_xlabel("Reference cluster")
ax.set_ylabel("% of generated sequences")
ax.set_title("Stage 2 generated-output cluster distribution\n(two independent 1000-sequence samples)")
ax.set_xticks(x + width / 2)
ax.set_xticklabels([f"Cluster {c}" for c in clusters_idx])
ax.legend()
plt.tight_layout()
plt.savefig("stage2_cluster_barplot.png", dpi=200)
print("saved stage2_cluster_barplot.png")

# --- UMAP: all embeddings from both samples, colored by cluster assignment ---
combined_embs = np.vstack(all_embeddings)
combined_labels = np.concatenate(all_labels)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
proj = reducer.fit_transform(combined_embs)

fig, ax = plt.subplots(figsize=(7, 6))
scatter = ax.scatter(proj[:, 0], proj[:, 1], c=combined_labels, cmap="tab10", s=8, alpha=0.6)
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
ax.add_artist(legend1)
ax.set_title("UMAP of Stage 2 generated sequences\n(ESM2 embedding space, colored by reference cluster)")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("stage2_umap.png", dpi=200)
print("saved stage2_umap.png")

# --- Reference: natural corpus cluster distribution, for comparison ---
ref_df = pd.read_csv("data/amp_clusters_assignments.csv")
ref_cluster_col = "cluster" if "cluster" in ref_df.columns else ref_df.columns[-1]
ref_counts = ref_df[ref_cluster_col].value_counts(normalize=True).sort_index() * 100
print("\nNatural corpus distribution (reference):")
print(ref_counts)
