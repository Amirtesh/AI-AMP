#!/usr/bin/env python3
"""
esm2_embedder.py
Shared ESM2 embedding extraction - loaded once, reused by model2_client.py
and ad_penalty.py, so both draw from the identical pipeline (esm2_t12_35M_UR50D,
layer 12, mean-pooled over valid residues) rather than risking two subtly
different implementations.
"""

import numpy as np
import torch
import esm

_model = None
_alphabet = None
_batch_converter = None
_device = None


def _ensure_loaded():
    global _model, _alphabet, _batch_converter, _device
    if _model is not None:
        return
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading esm2_t12_35M_UR50D on {_device} (one-time load)...")
    _model, _alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    _model = _model.to(_device).eval()
    _batch_converter = _alphabet.get_batch_converter()


def get_embeddings(sequences, batch_size=32):
    """Returns an (N, 480) array, mean-pooled layer-12 ESM2 embeddings,
    one row per input sequence, in input order. Identical logic to
    predict.py's get_embeddings(), just with the model persisted across
    calls instead of reloaded each time."""
    _ensure_loaded()
    all_emb = []
    for i in range(0, len(sequences), batch_size):
        batch = [(str(j), s) for j, s in enumerate(sequences[i:i + batch_size])]
        _, _, tokens = _batch_converter(batch)
        tokens = tokens.to(_device)
        with torch.no_grad():
            results = _model(tokens, repr_layers=[12])
        reps = results["representations"][12]
        for k, (_, s) in enumerate(batch):
            all_emb.append(reps[k, 1:len(s) + 1].mean(0).cpu().numpy())
    return np.vstack(all_emb)
