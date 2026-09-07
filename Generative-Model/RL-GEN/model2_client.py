#!/usr/bin/env python3
"""
model2_client.py
Local in-process Model 2 (AMP/non-AMP) scoring - loads the trained ESM2-only
classifier once, reuses the shared esm2_embedder for feature extraction.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import esm2_embedder

_clf = None
_esm2_cols = None
_threshold = None

SCRIPT_DIR = Path(__file__).resolve().parent


def _ensure_loaded():
    global _clf, _esm2_cols, _threshold
    if _clf is not None:
        return

    model_path = SCRIPT_DIR / "model2_esm2.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    _clf = joblib.load(model_path)

    with open(SCRIPT_DIR / "feature_columns.json") as f:
        cols = json.load(f)
    _esm2_cols = cols["esm2_cols"]

    ablation = pd.read_csv(SCRIPT_DIR / "ablation_results_with_thresholds.csv")
    thresh_rows = ablation[ablation["feature_set"] == "esm2"]
    if len(thresh_rows) == 0:
        raise RuntimeError("No 'esm2' row found in ablation_results_with_thresholds.csv")
    _threshold = float(thresh_rows["threshold"].iloc[0])
    print(f"Model 2 loaded. Decision threshold: {_threshold}")


def score_batch(sequences):
    """Returns a list of AMP-probability floats, one per input sequence, in
    the same order. Raises on any embedding/scoring failure rather than
    silently returning zeros - a broken reward signal should stop training
    loudly, not corrupt it quietly."""
    _ensure_loaded()
    if not sequences:
        return []

    emb_matrix = esm2_embedder.get_embeddings(sequences)
    if emb_matrix.shape[0] != len(sequences):
        raise RuntimeError(
            f"Embedding count mismatch: {emb_matrix.shape[0]} embeddings for "
            f"{len(sequences)} sequences - stop and debug before trusting reward."
        )

    X = pd.DataFrame(emb_matrix, columns=_esm2_cols)
    probs = _clf.predict_proba(X)[:, 1]
    return probs.tolist()


def get_threshold():
    _ensure_loaded()
    return _threshold
