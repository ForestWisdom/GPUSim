#!/usr/bin/env python3
"""Thin CLI wrapper around the training pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
    checkpoint = {
        "model_state_dict": result.model.state_dict(),
        "feature_columns": result.feature_columns,
        "feature_mean": result.feature_mean.detach().cpu(),
        "feature_std": result.feature_std.detach().cpu(),
        "hidden_sizes": result.hidden_sizes,
    }
    torch.save(checkpoint, args.checkpoint)
    print(f"final_train_loss={result.final_train_loss:.6f}")
    print(f"best_val_loss={result.best_val_loss:.6f}")


if __name__ == "__main__":
    main()
