#!/usr/bin/env python3
"""
Stage 1: Peptide-grammar pretraining on the combined PeptideAtlas corpus.

Usage:
    python3 train_stage1.py --data stage1_corpus_weighted.csv

Assumes stage1_corpus_weighted.csv (columns: sequence, organism, sample_weight)
is in the same directory unless --data points elsewhere. Checkpoints resume
automatically if interrupted - just rerun the same command.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Vocab (fixed, not a CLI arg - changing this would break checkpoint compat)
# --------------------------------------------------------------------------
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>"]
VOCAB = SPECIAL_TOKENS + AMINO_ACIDS
TOKEN2IDX = {tok: i for i, tok in enumerate(VOCAB)}
IDX2TOKEN = {i: tok for i, tok in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PAD_IDX = TOKEN2IDX["<PAD>"]
BOS_IDX = TOKEN2IDX["<BOS>"]
EOS_IDX = TOKEN2IDX["<EOS>"]


def encode_sequence(seq, max_len):
    ids = [BOS_IDX] + [TOKEN2IDX[c] for c in seq] + [EOS_IDX]
    pad_len = (max_len + 2) - len(ids)
    ids = ids + [PAD_IDX] * pad_len
    return ids


class WeightedAMPDataset(Dataset):
    def __init__(self, sequences, max_len):
        self.data = [encode_sequence(s, max_len) for s in sequences]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = torch.tensor(self.data[idx], dtype=torch.long)
        return ids[:-1], ids[1:]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class SmallPeptideGPT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=256, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            CausalSelfAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), diagonal=1)
        for block in self.blocks:
            x = block(x, causal_mask)
        x = self.ln_f(x)
        return self.head(x)


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------
def run_eval(model, val_loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1),
                ignore_index=PAD_IDX, reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += (yb != PAD_IDX).sum().item()
    model.train()
    return total_loss / total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="stage1_corpus_weighted.csv")
    ap.add_argument("--checkpoint_dir", default="./stage1_checkpoints")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", default=True,
                     help="Use mixed precision (bf16 if supported, else fp16)")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}, GPUs visible: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    use_bf16 = args.amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Mixed precision: {args.amp} ({'bf16' if use_bf16 else 'fp16' if args.amp else 'off'})")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Data ---
    df = pd.read_csv(args.data)
    df["sequence"] = df["sequence"].str.strip().str.upper()
    valid_mask = df["sequence"].apply(lambda s: set(s).issubset(set(AMINO_ACIDS)))
    n_invalid = (~valid_mask).sum()
    if n_invalid:
        print(f"Dropping {n_invalid} sequences with non-canonical residues")
    df = df[valid_mask].reset_index(drop=True)

    max_len = df["sequence"].str.len().max()
    print(f"Corpus: {len(df)} sequences, max length {max_len}")

    train_df, val_df = train_test_split(df, test_size=args.val_frac, random_state=args.seed)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    train_ds = WeightedAMPDataset(train_df["sequence"].tolist(), max_len)
    val_ds = WeightedAMPDataset(val_df["sequence"].tolist(), max_len)

    if "sample_weight" in train_df.columns:
        sampler = WeightedRandomSampler(
            weights=train_df["sample_weight"].tolist(),
            num_samples=len(train_df), replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                   num_workers=args.num_workers, pin_memory=True)
    else:
        print("No sample_weight column found - using plain shuffling")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=max(1, args.num_workers // 2), pin_memory=True)

    # --- Model ---
    model = SmallPeptideGPT(VOCAB_SIZE, max_len + 2, args.d_model, args.n_heads, args.n_layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if n_gpus > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and not use_bf16))

    # --- Resume ---
    start_epoch = 0
    best_val_loss = float("inf")
    ckpts = sorted(
        [f for f in os.listdir(args.checkpoint_dir) if f.startswith("epoch_")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    if ckpts:
        latest = ckpts[-1]
        print(f"Resuming from {latest}")
        ckpt = torch.load(os.path.join(args.checkpoint_dir, latest), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
    else:
        print("No checkpoint found, starting fresh")

    # --- Train loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss, running_tokens = 0.0, 0

        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
                logits = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1),
                                        ignore_index=PAD_IDX)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step()

            running_loss += loss.item() * xb.size(0)
            running_tokens += xb.size(0)

            if step % 200 == 0:
                elapsed = time.time() - epoch_start
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                      f"elapsed={elapsed/60:.1f}min")

        train_loss = running_loss / running_tokens
        val_loss = run_eval(model, val_loader, device)
        elapsed = time.time() - epoch_start

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"time={elapsed/60:.1f}min")

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }, os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt"))

        if is_best:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pt"))
            print(f"  -> new best model saved (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
