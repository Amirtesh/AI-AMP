"""
toxinpred3_server.py
Persistent local HTTP server wrapping ToxinPred3 Hybrid scoring.
Runs in the toxin_hemo conda env (sklearn ~1.2.2, required by the
bundled toxinpred3.0_model.pkl). Called by toxinpred3_client.py from
the RLGEN env, which cannot load this pickle directly (sklearn 1.3.1
raises ValueError on the Tree node dtype mismatch).

Start once, before GRPO training begins:
    conda activate toxin_hemo
    nohup python3 toxinpred3_server.py --port 8765 > toxinpred3_server.log 2>&1 &
    disown

All toxinpred3.py functions use RELATIVE filenames internally (no
working-directory parameter), so each request chdir's into a fresh
temp directory. Requests are processed sequentially (single lock) since
chdir is process-global and concurrent requests would collide.
"""

import os
import sys
import threading
import tempfile
import shutil

from flask import Flask, request, jsonify

TOXINPRED3_SCRIPTS = os.path.expanduser(
    "~/miniconda3/envs/toxin_hemo/lib/python3.10/site-packages/toxinpred3/python_scripts"
)
TOXINPRED3_ROOT = os.path.dirname(TOXINPRED3_SCRIPTS)

if not os.path.isdir(TOXINPRED3_SCRIPTS):
    raise RuntimeError(f"toxinpred3 scripts not found at {TOXINPRED3_SCRIPTS}")

sys.path.insert(0, TOXINPRED3_SCRIPTS)

import toxinpred3 as tp3  # noqa: E402

MERCI_BIN = os.path.join(TOXINPRED3_ROOT, "merci", "MERCI_motif_locator.pl")
MOTIFS_P = os.path.join(TOXINPRED3_ROOT, "motifs", "pos_motif.txt")
MOTIFS_N = os.path.join(TOXINPRED3_ROOT, "motifs", "neg_motif.txt")
MODEL_PKL = os.path.join(TOXINPRED3_ROOT, "model", "toxinpred3.0_model.pkl")

DEFAULT_THRESHOLD = 0.38

app = Flask(__name__)
_lock = threading.Lock()  # serialize requests since chdir is process-global


def _score_sequences(sequences, threshold=DEFAULT_THRESHOLD):
    orig_cwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp()
    try:
        os.chdir(tmp_dir)

        seqid = [f"seq_{i}" for i in range(len(sequences))]
        seqid_1 = [f">{s}" for s in seqid]

        import pandas as pd
        CM = pd.concat([pd.DataFrame(seqid_1), pd.DataFrame(sequences)], axis=1)
        CM.to_csv("Sequence_1", header=False, index=None, sep="\n")

        tp3.aac_comp(sequences, "seq.aac")
        os.system("perl -pi -e 's/,$//g' seq.aac")
        tp3.dpc_comp(sequences, "seq.dpc")
        os.system("perl -pi -e 's/,$//g' seq.dpc")
        tp3.prediction("seq.aac", "seq.dpc", MODEL_PKL, "seq.pred")

        os.system(f"perl {MERCI_BIN} -p Sequence_1 -i {MOTIFS_P} -o merci_p.txt")
        os.system(f"perl {MERCI_BIN} -p Sequence_1 -i {MOTIFS_N} -o merci_n.txt")
        tp3.MERCI_Processor_p("merci_p.txt", "merci_output_p.csv", seqid)
        tp3.Merci_after_processing_p("merci_output_p.csv", "merci_hybrid_p.csv")
        tp3.MERCI_Processor_n("merci_n.txt", "merci_output_n.csv", seqid)
        tp3.Merci_after_processing_n("merci_output_n.csv", "merci_hybrid_n.csv")

        tp3.hybrid("seq.pred", seqid, "merci_hybrid_p.csv", "merci_hybrid_n.csv", threshold, "final_output")

        df = pd.read_csv("final_output")
        df.loc[df["Hybrid Score"] > 1, "Hybrid Score"] = 1
        df.loc[df["Hybrid Score"] < 0, "Hybrid Score"] = 0

        df["SeqID"] = df.iloc[:, 0].astype(str)
        score_by_id = dict(zip(df["SeqID"], df["Hybrid Score"]))
        scores = [float(score_by_id.get(sid, 0.0)) for sid in seqid]
        return scores
    finally:
        os.chdir(orig_cwd)
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/score", methods=["POST"])
def score():
    data = request.get_json()
    sequences = data.get("sequences", [])
    threshold = data.get("threshold", DEFAULT_THRESHOLD)
    if not sequences:
        return jsonify({"scores": []})
    with _lock:
        scores = _score_sequences(sequences, threshold)
    return jsonify({"scores": scores})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"ToxinPred3 scoring server starting on port {args.port}...")
    app.run(host="127.0.0.1", port=args.port, threaded=False)
