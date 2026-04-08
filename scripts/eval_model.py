#!/usr/bin/env python3
"""Evaluate a checkpoint on a processed dataset."""

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

from perf_model.model.mlp import LatencyMLP
from perf_model.pipelines.eval_pipeline import evaluate_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.dataset)
    payload = torch.load(args.checkpoint, map_location="cpu")
    feature_columns = [column for column in frame.columns if column.startswith("f_")]
    if isinstance(payload, dict) and "model_state_dict" in payload:
        hidden_sizes = payload.get("hidden_sizes")
        model = LatencyMLP(input_dim=len(feature_columns), hidden_sizes=hidden_sizes)
        model.load_state_dict(payload["model_state_dict"])
        feature_mean = payload.get("feature_mean")
        feature_std = payload.get("feature_std")
    else:
        model = LatencyMLP(input_dim=len(feature_columns))
        model.load_state_dict(payload)
        feature_mean = None
        feature_std = None
    metrics = evaluate_frame(model, frame, feature_mean=feature_mean, feature_std=feature_std)
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
