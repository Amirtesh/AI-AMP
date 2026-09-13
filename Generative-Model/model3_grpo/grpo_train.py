#!/usr/bin/env python3
"""
grpo_train.py
GRPO training for peptide generation - GRPO with composite reward
(Model2 gatekeeper score, physicochemical shaping, class-diversity term)
and CLI-overridable weights for interactive experimentation.
decode_sequences() stops at any non-amino-acid token, preventing the
special-token leak that crashed early development runs.
"""

import argparse
import copy
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F

from train_stage1 import SmallPeptideGPT, VOCAB_SIZE, IDX2TOKEN, PAD_IDX, BOS_IDX, EOS_IDX, AMINO_ACIDS
from reward import compute_rewards
import class_diversity

STAGE1_MAX_LEN = 83


def load_model(checkpoint_path, device):
    model = SmallPeptideGPT(VOCAB_SIZE, STAGE1_MAX_LEN + 2, d_model=256, n_heads=8, n_layers=6)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    return model


def sample_group_with_logprobs(model, device, group_size, min_len, max_len, temperature=1.0):
    hard_cap = max_len if max_len is not None else STAGE1_MAX_LEN
    ids = torch.full((group_size, 1), BOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(group_size, dtype=torch.bool, device=device)
    token_logprobs = []
    token_mask = []

    for _ in range(hard_cap):
        logits = model(ids)[:, -1, :] / temperature
        cur_len = ids.shape[1] - 1
        if min_len is not None and cur_len < min_len:
            logits = logits.clone()
            logits[:, EOS_IDX] = float("-inf")

        logits = logits.clone()
        logits[:, BOS_IDX] = float("-inf")
        logits[:, PAD_IDX] = float("-inf")
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        next_ids = torch.multinomial(probs, 1).squeeze(-1)

        step_logprob = log_probs.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
        step_mask = (~finished).float()

        next_ids = torch.where(finished, torch.full_like(next_ids, PAD_IDX), next_ids)
        finished = finished | (next_ids == EOS_IDX)

        token_logprobs.append(step_logprob)
        token_mask.append(step_mask)
        ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)

        if finished.all():
            break

    token_logprobs = torch.stack(token_logprobs, dim=1)
    token_mask = torch.stack(token_mask, dim=1)
    return ids, token_logprobs, token_mask


def decode_sequences(ids):
    """Includes the defensive valid_chars fix from the start - stops at ANY
    non-amino-acid token, not just EOS/PAD, preventing the '<' special-token
    leak that crashed the original run at step 31."""
    valid_chars = set(AMINO_ACIDS)
    sequences = []
    for row in ids.tolist():
        chars = []
        for tok in row[1:]:
            if tok in (EOS_IDX, PAD_IDX):
                break
            c = IDX2TOKEN[tok]
            if c not in valid_chars:
                break
            chars.append(c)
        sequences.append("".join(chars))
    return sequences


def reference_logprobs(ref_model, ids, token_mask):
    with torch.no_grad():
        logits = ref_model(ids[:, :-1])
        log_probs_all = F.log_softmax(logits, dim=-1)
        target_ids = ids[:, 1:]
        ref_logprobs = log_probs_all.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    T = token_mask.shape[1]
    return ref_logprobs[:, :T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_checkpoint", default="stage2_checkpoints/best_model.pt")
    ap.add_argument("--checkpoint_dir", default="./grpo_checkpoints")
    ap.add_argument("--cluster_kmeans", default="amp_clusters_kmeans_k5.joblib")
    ap.add_argument("--cluster_assignments", default="amp_clusters_assignments.csv")
    ap.add_argument("--training_seq", default="training_seq.csv")
    ap.add_argument("--group_size", type=int, default=256)
    ap.add_argument("--total_steps", type=int, default=500)
    ap.add_argument("--min_len", type=int, default=15)
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--w_model2", type=float, default=None, help="override WEIGHTS[model2]")
    ap.add_argument("--w_charge", type=float, default=None, help="override WEIGHTS[charge]")
    ap.add_argument("--w_length", type=float, default=None, help="override WEIGHTS[length]")
    ap.add_argument("--w_hmom", type=float, default=None, help="override WEIGHTS[hydrophobic_moment]")
    ap.add_argument("--w_diversity_penalty", type=float, default=None, help="override WEIGHTS[diversity_penalty]")
    ap.add_argument("--w_class_diversity", type=float, default=None, help="override WEIGHTS[class_diversity]")
    ap.add_argument("--w_hemolysis_penalty", type=float, default=None, help="override WEIGHTS[hemolysis_penalty]")
    ap.add_argument("--w_toxicity_penalty", type=float, default=None, help="override WEIGHTS[toxicity_penalty]")
    args = ap.parse_args()

    weight_overrides = {}
    if args.w_model2 is not None:
        weight_overrides["model2"] = args.w_model2
    if args.w_charge is not None:
        weight_overrides["charge"] = args.w_charge
    if args.w_length is not None:
        weight_overrides["length"] = args.w_length
    if args.w_hmom is not None:
        weight_overrides["hydrophobic_moment"] = args.w_hmom
    if args.w_diversity_penalty is not None:
        weight_overrides["diversity_penalty"] = args.w_diversity_penalty
    if args.w_class_diversity is not None:
        weight_overrides["class_diversity"] = args.w_class_diversity
    if args.w_hemolysis_penalty is not None:
        weight_overrides["hemolysis_penalty"] = args.w_hemolysis_penalty
    if args.w_toxicity_penalty is not None:
        weight_overrides["toxicity_penalty"] = args.w_toxicity_penalty
    if weight_overrides:
        print(f"Weight overrides active: {weight_overrides}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    class_diversity.load_cluster_model(args.cluster_kmeans, args.cluster_assignments)

    policy = load_model(args.init_checkpoint, device)
    policy.train()

    reference = copy.deepcopy(policy)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    training_seqs = set(pd.read_csv(args.training_seq)["sequence"].astype(str).str.strip().str.upper())
    compounds_log_path = os.path.join(args.checkpoint_dir, "compounds_generated.csv")
    seen_novel = set()
    file_is_new = not os.path.exists(compounds_log_path)
    if not file_is_new:
        existing = pd.read_csv(compounds_log_path)
        seen_novel = set(existing["sequence"].tolist())
        print(f"Resuming compound log: {len(seen_novel)} novel sequences already recorded")
    compounds_file = open(compounds_log_path, "a", buffering=1)
    if file_is_new:
        compounds_file.write("step,sequence,reward,cluster_id\n")

    start_step = 0
    ckpts = sorted(
        [f for f in os.listdir(args.checkpoint_dir) if f.startswith("step_")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    if ckpts:
        latest = ckpts[-1]
        print(f"Resuming from {latest}")
        ckpt = torch.load(os.path.join(args.checkpoint_dir, latest), map_location=device)
        policy.load_state_dict(ckpt["policy_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"] + 1
    else:
        print("No GRPO checkpoint found, starting fresh from Stage 2 model")

    for step in range(start_step, args.total_steps):
        step_start = time.time()

        ids, policy_logprobs, token_mask = sample_group_with_logprobs(
            policy, device, args.group_size, args.min_len, args.max_len, args.temperature
        )
        sequences = decode_sequences(ids)

        rewards, breakdown = compute_rewards(sequences, weights=weight_overrides or None)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)

        for seq, r, cid in zip(sequences, rewards, breakdown["cluster_id"]):
            if seq not in training_seqs and seq not in seen_novel:
                seen_novel.add(seq)
                compounds_file.write(f"{step},{seq},{r:.4f},{cid}\n")

        mean_r = rewards_t.mean()
        std_r = rewards_t.std().clamp_min(1e-6)
        advantages = (rewards_t - mean_r) / std_r

        ref_logprobs = reference_logprobs(reference, ids, token_mask)

        valid_tokens = token_mask.sum(dim=1).clamp_min(1.0)
        seq_logprob_mean = (policy_logprobs * token_mask).sum(dim=1) / valid_tokens
        pg_loss = -(advantages * seq_logprob_mean).mean()

        log_ratio = ref_logprobs - policy_logprobs
        kl_per_token = torch.exp(log_ratio) - log_ratio - 1
        kl_loss = (kl_per_token * token_mask).sum(dim=1) / valid_tokens
        kl_loss = kl_loss.mean()

        total_loss = pg_loss + args.kl_coef * kl_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()

        elapsed = time.time() - step_start

        if step % args.log_every == 0:
            print(f"step {step}/{args.total_steps} "
                  f"mean_reward={mean_r.item():.3f} std={std_r.item():.3f} "
                  f"pg_loss={pg_loss.item():.4f} kl={kl_loss.item():.4f} "
                  f"total_loss={total_loss.item():.4f} "
                  f"time={elapsed:.1f}s")
            print(f"  reward terms (mean): "
                  f"model2={sum(breakdown['model2'])/len(breakdown['model2']):.3f} "
                  f"charge={sum(breakdown['charge'])/len(breakdown['charge']):.3f} "
                  f"length={sum(breakdown['length'])/len(breakdown['length']):.3f} "
                  f"hmom={sum(breakdown['hydrophobic_moment'])/len(breakdown['hydrophobic_moment']):.3f} "
                  f"div={sum(breakdown['diversity_penalty'])/len(breakdown['diversity_penalty']):.3f} "
                  f"class_div={sum(breakdown['class_diversity'])/len(breakdown['class_diversity']):.3f}")
            cluster_counts = pd.Series(breakdown["cluster_id"]).value_counts().sort_index()
            print(f"  cluster distribution this step: {cluster_counts.to_dict()}")

        if step % args.save_every == 0 or step == args.total_steps - 1:
            torch.save({
                "step": step,
                "policy_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "mean_reward": mean_r.item(),
            }, os.path.join(args.checkpoint_dir, f"step_{step}.pt"))
            print(f"  -> checkpoint saved at step {step}")


if __name__ == "__main__":
    main()
