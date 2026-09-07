#!/usr/bin/env python3
"""Quick diagnostic on the trained Stage 1 model: real-vs-scrambled loss
comparison, plus sample generation to eyeball output quality."""

import torch
import torch.nn.functional as F
import random

from train_stage1 import SmallPeptideGPT, VOCAB, VOCAB_SIZE, TOKEN2IDX, IDX2TOKEN, \
    PAD_IDX, BOS_IDX, EOS_IDX, AMINO_ACIDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model - match architecture args used during training
MAX_LEN = 83  # from training run: "Corpus: 3374346 sequences, max length 83"
model = SmallPeptideGPT(VOCAB_SIZE, MAX_LEN + 2, d_model=256, n_heads=8, n_layers=6)
state_dict = torch.load("stage1_checkpoints/best_model.pt", map_location=device)
model.load_state_dict(state_dict)
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

# --- Real vs scrambled, several examples ---
test_seqs = [
    "GIGKFLHSAKKFGKAFVGEIMNS",   # magainin-like
    "MKTAYIAKQRQISFVKSHFSRQLEE",  # generic protein fragment
    "FLPLIGRVLSGIL",              # short AMP-like
]

print("=== Real vs. scrambled loss comparison ===")
for seq in test_seqs:
    scrambled = "".join(random.sample(seq, len(seq)))
    real_loss = sequence_loss(seq)
    scr_loss = sequence_loss(scrambled)
    print(f"real: {seq}\n  loss={real_loss:.4f}")
    print(f"scr:  {scrambled}\n  loss={scr_loss:.4f}  (gap: {scr_loss - real_loss:+.4f})\n")

# --- Sample generation ---
@torch.no_grad()
def generate(max_new_tokens=60, temperature=1.0, top_p=0.9):
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

print("=== Sample generations ===")
for _ in range(10):
    print(generate())
