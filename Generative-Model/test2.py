#!/usr/bin/env python3
"""Stage 2 diagnostic: real-vs-scrambled on AMP-like sequences (should NOW
show a real gap, unlike Stage 1), composition sanity check on generated
output, and a memorization check against the training corpus."""

import torch
import torch.nn.functional as F
import random
import pandas as pd

from train_stage1 import SmallPeptideGPT, VOCAB_SIZE, TOKEN2IDX, IDX2TOKEN, \
    PAD_IDX, BOS_IDX, EOS_IDX, AMINO_ACIDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STAGE1_MAX_LEN = 83  # must match training - positional embedding size
model = SmallPeptideGPT(VOCAB_SIZE, STAGE1_MAX_LEN + 2, d_model=256, n_heads=8, n_layers=6)
model.load_state_dict(torch.load("stage2_checkpoints/best_model.pt", map_location=device))
model.to(device)
model.eval()

def sequence_loss(seq):
    ids = [BOS_IDX] + [TOKEN2IDX[c] for c in seq] + [EOS_IDX]
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
    return loss.item()

# Real, well-known AMPs NOT necessarily identical to training entries -
# magainin-2, cecropin-fragment-like, LL-37-fragment-like
test_seqs = [
    "GIGKFLHSAKKFGKAFVGEIMNS",   # magainin-2
    "KWKLFKKIGAVLKVL",           # cecropin-like
    "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",  # LL-37
]

print("=== Real AMP vs scrambled loss (Stage 2 model) ===")
for seq in test_seqs:
    scrambled = "".join(random.sample(seq, len(seq)))
    real_loss = sequence_loss(seq)
    scr_loss = sequence_loss(scrambled)
    print(f"real: {seq}\n  loss={real_loss:.4f}")
    print(f"scr:  {scrambled}\n  loss={scr_loss:.4f}  (gap: {scr_loss - real_loss:+.4f})\n")

@torch.no_grad()
def generate(max_new_tokens=50, temperature=1.0, top_p=0.9):
    ids = [BOS_IDX]
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], device=device)
        logits = model(x)[0, -1] / temperature
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

def net_charge(seq):
    pos = seq.count("K") + seq.count("R") + 0.1 * seq.count("H")
    neg = seq.count("D") + seq.count("E")
    return pos - neg

print("=== Sample generations + composition check ===")
samples = [generate() for _ in range(20)]
for s in samples:
    print(f"{s}  (len={len(s)}, net_charge={net_charge(s):.1f})")

# --- Memorization check: exact/near-duplicate against training corpus ---
train_df = pd.read_csv("amp_positive_filtered.csv")
train_seqs = set(train_df["sequence"].astype(str).str.strip().str.upper())
exact_matches = sum(1 for s in samples if s in train_seqs)
print(f"\nExact matches to training corpus: {exact_matches}/{len(samples)}")
