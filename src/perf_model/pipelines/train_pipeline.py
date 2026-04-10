"""Dataset-to-model training pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from perf_model.features.feature_vector import get_feature_column_name
from perf_model.model.loss import build_loss
from perf_model.model.metrics import mape, rmse
from perf_model.model.mlp import LatencyMLP
from perf_model.model.predict import reconstruct_latencies


@dataclass(slots=True)
class TrainResult:
    model: LatencyMLP
    final_train_loss: float
    best_val_loss: float
    best_epoch: int
    feature_columns: list[str]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    hidden_sizes: list[int]
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    target_kind: str
    theoretical_cycle_feature: str
    loss_name: str
    dropout: float
    use_batch_norm: bool


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


def _regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    preds_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    return {
        "mape": mape(targets_np, preds_np),
        "rmse": rmse(targets_np, preds_np),
    }


def _build_efficiency_target(
    frame: pd.DataFrame,
    *,
    theoretical_cycle_feature: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theoretical_cycles = torch.tensor(
        frame[theoretical_cycle_feature].to_numpy(), dtype=torch.float32
    )
    clock_column = get_feature_column_name("gpu_clock_mhz")
    clock_mhz = torch.tensor(frame[clock_column].to_numpy(), dtype=torch.float32)
    actual_cycles = torch.tensor(frame["latency_us"].to_numpy(), dtype=torch.float32) * clock_mhz
    efficiency = torch.clamp(
        theoretical_cycles / torch.clamp(actual_cycles, min=1e-6),
        min=0.0,
        max=1.0,
    )
    return efficiency, theoretical_cycles, clock_mhz


def train_from_frame(
    frame: pd.DataFrame,
    hidden_sizes: list[int] | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    train_ratio: float = 0.8,
    patience: int = 20,
    loss_name: str = "mape",
    target_kind: str = "efficiency",
    dropout: float = 0.1,
    use_batch_norm: bool = True,
) -> TrainResult:
    feature_columns = [column for column in frame.columns if column.startswith("f_")]
    if not feature_columns:
        raise ValueError("dataset does not contain feature columns")
    if "latency_us" not in frame.columns:
        raise ValueError("dataset does not contain latency_us")
    if target_kind != "efficiency":
        raise ValueError(f"unsupported target_kind: {target_kind}")

    theoretical_cycle_feature = get_feature_column_name("max_sm_busy_cycles")
    if theoretical_cycle_feature not in frame.columns:
        raise ValueError(f"dataset does not contain {theoretical_cycle_feature}")

    train_df, val_df = _split_frame(frame, train_ratio=train_ratio)

    x_train, y_train_latency, mean, std = _prepare_tensors(train_df, feature_columns)
    x_val, y_val_latency, _, _ = _prepare_tensors(val_df, feature_columns, mean=mean, std=std)
    y_train, train_theoretical_cycles, train_clock_mhz = _build_efficiency_target(
        train_df, theoretical_cycle_feature=theoretical_cycle_feature
    )
    y_val, val_theoretical_cycles, val_clock_mhz = _build_efficiency_target(
        val_df, theoretical_cycle_feature=theoretical_cycle_feature
    )

    resolved_hidden_sizes = hidden_sizes or [256, 256, 128]
    model = LatencyMLP(
        input_dim=len(feature_columns),
        hidden_sizes=resolved_hidden_sizes,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    )
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

    model.eval()
    with torch.no_grad():
        best_train_pred = model(x_train)
        best_val_pred = model(x_val)
    train_latency_pred = reconstruct_latencies(best_train_pred, train_theoretical_cycles, train_clock_mhz)
    val_latency_pred = reconstruct_latencies(best_val_pred, val_theoretical_cycles, val_clock_mhz)

    train_metrics = _regression_metrics(train_latency_pred, y_train_latency)
    val_metrics = _regression_metrics(val_latency_pred, y_val_latency)

    return TrainResult(
        model=model,
        final_train_loss=last_train_loss,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        feature_columns=feature_columns,
        feature_mean=mean,
        feature_std=std,
        hidden_sizes=resolved_hidden_sizes,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        target_kind=target_kind,
        theoretical_cycle_feature=theoretical_cycle_feature,
        loss_name=loss_name,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    )
