import joblib, pandas as pd, numpy as np
from esm2_embedder import get_embeddings

km = joblib.load('amp_clusters_kmeans_k5.joblib')
df = pd.read_csv('grpo_dual_toxicity_trial_checkpoints/compounds_generated.csv')
recent = df[df['step'] >= 75]['sequence'].dropna().unique().tolist()

embs = get_embeddings(recent)
clusters = km.predict(embs)
vals, counts = np.unique(clusters, return_counts=True)
for v, c in zip(vals, counts):
    print(f'cluster {v}: {c} ({c/len(clusters)*100:.1f}%)')
