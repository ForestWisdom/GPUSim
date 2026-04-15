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
    parser.add_argument("--loss", default="mape")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--launch-overhead-us", type=float, default=0.0)
    parser.add_argument("--checkpoint", required=True, help="Output checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.train)
    result = train_from_frame(
        frame,
        epochs=args.epochs,
        lr=args.lr,
        loss_name=args.loss,
        dropout=args.dropout,
        launch_overhead_us=args.launch_overhead_us,
    )
    checkpoint = {
        "model_state_dict": result.model.state_dict(),
        "feature_columns": result.feature_columns,
        "feature_mean": result.feature_mean.detach().cpu(),
        "feature_std": result.feature_std.detach().cpu(),
        "hidden_sizes": result.hidden_sizes,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "train_metrics": result.train_metrics,
        "val_metrics": result.val_metrics,
        "target_kind": result.target_kind,
        "theoretical_cycle_feature": result.theoretical_cycle_feature,
        "loss_name": result.loss_name,
        "dropout": result.dropout,
        "use_batch_norm": result.use_batch_norm,
        "launch_overhead_us": result.launch_overhead_us,
    }
    torch.save(checkpoint, args.checkpoint)
    print(f"final_train_loss={result.final_train_loss:.6f}")
    print(f"best_epoch={result.best_epoch}")
    print(f"best_val_loss={result.best_val_loss:.6f}")
    print(f"val_rmse={result.val_metrics['rmse']:.6f}")
    print(f"val_mape={result.val_metrics['mape']:.6f}")


if __name__ == "__main__":
    main()
