#!/usr/bin/env python3
"""
track_compounds.py
Live progress monitor for a running GRPO job - polls the compounds log
and prints growing counts + reward trend without touching the training
process itself.

Usage:
    python3 track_compounds.py --checkpoint_dir grpo_checkpoints --interval 30
"""

import argparse
import os
import time

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default="grpo_checkpoints")
    ap.add_argument("--interval", type=int, default=30, help="Seconds between checks")
    ap.add_argument("--once", action="store_true", help="Print once and exit, no polling loop")
    args = ap.parse_args()

    log_path = os.path.join(args.checkpoint_dir, "compounds_generated.csv")

    last_count = 0
    start_time = time.time()

    while True:
        if not os.path.exists(log_path):
            print(f"Waiting for {log_path} to be created...")
        else:
            df = pd.read_csv(log_path)
            count = len(df)
            elapsed_min = (time.time() - start_time) / 60
            rate = (count - last_count) / max(args.interval / 60, 1e-6) if last_count else 0

            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"novel compounds so far: {count} | "
                  f"latest step: {df['step'].max() if count else '-'} | "
                  f"mean reward (last 100): {df['reward'].tail(100).mean():.3f}" if count else
                  f"[{time.strftime('%H:%M:%S')}] no compounds logged yet")

            last_count = count

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
