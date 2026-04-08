#!/usr/bin/env python3
"""Thin CLI wrapper around the training pipeline."""

from __future__ import annotations

import argparse

import pandas as pd
import torch

from perf_model.pipelines.train_pipeline import train_from_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Processed CSV path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", required=True, help="Output checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.train)
    result = train_from_frame(frame, epochs=args.epochs, lr=args.lr)
    torch.save(result.model.state_dict(), args.checkpoint)
    print(f"final_loss={result.final_loss:.6f}")


if __name__ == "__main__":
    main()
