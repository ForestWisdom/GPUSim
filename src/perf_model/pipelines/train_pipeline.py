"""Dataset-to-model training pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from perf_model.model.loss import build_loss
from perf_model.model.mlp import LatencyMLP


@dataclass(slots=True)
class TrainResult:
    model: LatencyMLP
    final_train_loss: float
    best_val_loss: float
    feature_columns: list[str]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    hidden_sizes: list[int]


def _split_frame(frame: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = max(1, int(len(shuffled) * train_ratio))
    train_df = shuffled.iloc[:split_idx].copy()
    val_df = shuffled.iloc[split_idx:].copy()
    if len(val_df) == 0:
        val_df = train_df.copy()
    return train_df, val_df


def _prepare_tensors(
    frame: pd.DataFrame,
    feature_columns: list[str],
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(frame[feature_columns].to_numpy(), dtype=torch.float32)
    y = torch.tensor(frame["latency_us"].to_numpy(), dtype=torch.float32)

    if mean is None:
        mean = x.mean(dim=0)
    if std is None:
        std = x.std(dim=0)

    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    x_norm = (x - mean) / std
    return x_norm, y, mean, std


def train_from_frame(
    frame: pd.DataFrame,
    hidden_sizes: list[int] | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    train_ratio: float = 0.8,
    patience: int = 20,
    loss_name: str = "mse",
) -> TrainResult:
    feature_columns = [column for column in frame.columns if column.startswith("f_")]
    if not feature_columns:
        raise ValueError("dataset does not contain feature columns")
    if "latency_us" not in frame.columns:
        raise ValueError("dataset does not contain latency_us")

    train_df, val_df = _split_frame(frame, train_ratio=train_ratio)

    x_train, y_train, mean, std = _prepare_tensors(train_df, feature_columns)
    x_val, y_val, _, _ = _prepare_tensors(val_df, feature_columns, mean=mean, std=std)

    resolved_hidden_sizes = hidden_sizes or [256, 256, 128]
    model = LatencyMLP(input_dim=len(feature_columns), hidden_sizes=resolved_hidden_sizes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = build_loss(loss_name)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    last_train_loss = 0.0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pred_train = model(x_train)
        train_loss = loss_fn(pred_train, y_train)
        train_loss.backward()
        optimizer.step()

        last_train_loss = float(train_loss.item())

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val)
            val_loss = float(loss_fn(pred_val, y_val).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        final_train_loss=last_train_loss,
        best_val_loss=best_val_loss,
        feature_columns=feature_columns,
        feature_mean=mean,
        feature_std=std,
        hidden_sizes=resolved_hidden_sizes,
    )
