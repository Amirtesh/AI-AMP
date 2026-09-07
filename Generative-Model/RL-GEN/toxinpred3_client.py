"""
toxinpred3_client.py
Calls the persistent toxinpred3_server.py (running in toxin_hemo env,
started separately via nohup) over HTTP. Mirrors hemopi2_client.py's
score_batch() interface so reward.py integration follows the same
pattern for both toxicity terms.
"""

import requests

SERVER_URL = "http://127.0.0.1:8765"
DEFAULT_THRESHOLD = 0.38


def score_batch(sequences: list, threshold: float = DEFAULT_THRESHOLD) -> list:
    if not sequences:
        return []

    try:
        resp = requests.post(
            f"{SERVER_URL}/score",
            json={"sequences": sequences, "threshold": threshold},
            timeout=60,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "toxinpred3_server.py is not running. Start it first:\n"
            "  conda activate toxin_hemo\n"
            "  nohup python3 toxinpred3_server.py --port 8765 > toxinpred3_server.log 2>&1 &\n"
            "  disown"
        )

    resp.raise_for_status()
    return resp.json()["scores"]


if __name__ == "__main__":
    test_seqs = [
        "GIGKKILRIGKILKNLFKGIGK",
        "ACDEFGHIKLMNPQRSTVWY",
    ]
    scores = score_batch(test_seqs)
    for seq, score in zip(test_seqs, scores):
        print(f"{score:.3f}  {seq}")
