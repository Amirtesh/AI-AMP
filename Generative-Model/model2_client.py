#!/usr/bin/env python3
"""
model2_client.py
HTTP client for the Model 2 (AMP gatekeeper) Flask scoring server.

*** INTEGRATION POINT - CONFIRM/EDIT THESE TWO THINGS ***
1. MODEL2_URL below - point at your actual running server.
2. The request/response format in score_batch() - written assuming a
   batched POST endpoint accepting {"sequences": [...]} and returning
   {"probabilities": [...]}. If your server's actual contract differs
   (single-sequence only, different field names), this needs editing
   before anything else in this pipeline will work.
"""

import requests

MODEL2_URL = "http://localhost:5000/predict"  # EDIT to match your server
TIMEOUT_SECONDS = 60


def score_batch(sequences):
    """Returns a list of AMP-probability floats, one per input sequence,
    in the same order. Raises on request failure rather than silently
    returning zeros - a scoring failure should stop training, not
    quietly corrupt the reward signal."""
    if not sequences:
        return []
    resp = requests.post(
        MODEL2_URL,
        json={"sequences": sequences},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    probs = data["probabilities"]
    if len(probs) != len(sequences):
        raise RuntimeError(
            f"Model 2 server returned {len(probs)} scores for {len(sequences)} "
            f"sequences - response format mismatch, check the contract."
        )
    return probs
