"""Dataset-to-model training pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from perf_model.model.loss import build_loss
from perf_model.model.mlp import LatencyMLP
from perf_model.model.train import train_epoch


@dataclass(slots=True)
class TrainResult:
    model: LatencyMLP
    final_loss: float


def train_from_frame(
    frame: pd.DataFrame,
    hidden_sizes: list[int] | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
) -> TrainResult:
    feature_columns = [column for column in frame.columns if column.startswith("f_")]
    if not feature_columns:
        raise ValueError("dataset does not contain feature columns")
    features = torch.tensor(frame[feature_columns].to_numpy(), dtype=torch.float32)
    targets = torch.tensor(frame["latency_us"].to_numpy(), dtype=torch.float32)
    model = LatencyMLP(input_dim=len(feature_columns), hidden_sizes=hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = build_loss("mse")
    last_loss = 0.0
    for _ in range(epochs):
        last_loss = train_epoch(model, optimizer, loss_fn, features, targets)
    return TrainResult(model=model, final_loss=last_loss)
