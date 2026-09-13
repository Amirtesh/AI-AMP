#!/usr/bin/env python3
"""
Stage 2: Fine-tune the Stage 1 peptide-grammar model on AMP-positive
sequences (15-50aa), shifting it toward AMP-like composition/motifs.

Usage:
    python3 train_stage2.py --data amp_positive_filtered.csv \
        --init_checkpoint stage1_checkpoints/best_model.pt
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from train_stage1 import (
    SmallPeptideGPT, WeightedAMPDataset, VOCAB_SIZE, PAD_IDX,
    AMINO_ACIDS, IDX2TOKEN, run_eval,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="amp_positive_filtered.csv")
    ap.add_argument("--seq_column", default="sequence")
    ap.add_argument("--init_checkpoint", default="stage1_checkpoints/best_model.pt")
    ap.add_argument("--checkpoint_dir", default="./stage2_checkpoints")
    ap.add_argument("--min_len", type=int, default=15)
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    # Fine-tuning LR should be meaningfully lower than Stage 1's 6e-4 - we're
    # adapting an already-trained model, not learning from scratch. Too high
    # a LR here risks catastrophic forgetting of Stage 1's general grammar.
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--stage1_max_len", type=int, default=83,
                     help="MUST match the max_len used to train the Stage 1 "
                          "checkpoint being loaded - determines positional "
                          "embedding table size.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    use_bf16 = args.amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Mixed precision: {args.amp} ({'bf16' if use_bf16 else 'fp16' if args.amp else 'off'})")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Data ---
    df = pd.read_csv(args.data)
    df[args.seq_column] = df[args.seq_column].astype(str).str.strip().str.upper()

    valid_mask = df[args.seq_column].apply(lambda s: set(s).issubset(set(AMINO_ACIDS)))
    n_invalid = (~valid_mask).sum()
    if n_invalid:
        print(f"Dropping {n_invalid} sequences with non-canonical residues")
    df = df[valid_mask]

    df["seq_len"] = df[args.seq_column].str.len()
    before = len(df)
    df = df[(df["seq_len"] >= args.min_len) & (df["seq_len"] <= args.max_len)]
    print(f"Length filter [{args.min_len}-{args.max_len}]: {before} -> {len(df)}")

    before = len(df)
    df = df.drop_duplicates(subset=[args.seq_column]).reset_index(drop=True)
    print(f"Dedup: {before} -> {len(df)}")

    print(f"Final Stage 2 corpus: {len(df)} sequences")
    print(df["seq_len"].describe())

    if len(df) < 1000:
        print("WARNING: fewer than 1000 sequences after filtering - double-check "
              "--data and --seq_column point to the right file/column before proceeding.")

    train_df, val_df = train_test_split(df, test_size=args.val_frac, random_state=args.seed)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # NOTE: max_len for encoding must match Stage 1's positional embedding
    # size (args.stage1_max_len), NOT this stage's shorter 15-50 range -
    # the model's pos_emb table was sized for Stage 1's data and can't be
    # silently resized without losing the loaded weights for those positions.
    train_ds = WeightedAMPDataset(train_df[args.seq_column].tolist(), args.stage1_max_len)
    val_ds = WeightedAMPDataset(val_df[args.seq_column].tolist(), args.stage1_max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=max(1, args.num_workers // 2), pin_memory=True)

    # --- Model: same architecture, load Stage 1 weights ---
    model = SmallPeptideGPT(VOCAB_SIZE, args.stage1_max_len + 2,
                             args.d_model, args.n_heads, args.n_layers)
    state_dict = torch.load(args.init_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded Stage 1 weights from {args.init_checkpoint}")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and not use_bf16))

    start_epoch = 0
    best_val_loss = float("inf")
    ckpts = sorted(
        [f for f in os.listdir(args.checkpoint_dir) if f.startswith("epoch_")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    if ckpts:
        latest = ckpts[-1]
        print(f"Resuming Stage 2 from {latest}")
        ckpt = torch.load(os.path.join(args.checkpoint_dir, latest), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]

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

            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss = running_loss / running_tokens
        val_loss = run_eval(model, val_loader, device)
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"time={elapsed/60:.2f}min")

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss": val_loss, "best_val_loss": best_val_loss,
        }, os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt"))

        if is_best:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pt"))
            print(f"  -> new best model saved (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
