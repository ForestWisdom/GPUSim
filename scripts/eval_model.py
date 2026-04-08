#!/usr/bin/env python3
"""Evaluate a checkpoint on a processed dataset."""

from __future__ import annotations

import argparse

import pandas as pd
import torch

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
    feature_columns = [column for column in frame.columns if column.startswith("f_")]
    model = LatencyMLP(input_dim=len(feature_columns))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    metrics = evaluate_frame(model, frame)
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
