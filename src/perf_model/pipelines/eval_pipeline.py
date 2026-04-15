"""Evaluation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from perf_model.features.feature_vector import get_feature_column_name
from perf_model.model.metrics import mape, percentile_relative_error, rmse
from perf_model.model.predict import predict_latencies


def evaluate_frame(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
    *,
    target_kind: str = "latency",
    theoretical_cycle_feature: str = "f_33",
    launch_overhead_us: float = 0.0,
) -> dict[str, float]:
    feature_columns = [column for column in frame.columns if column.startswith("f_")]
    features = torch.tensor(frame[feature_columns].to_numpy(), dtype=torch.float32)
    if feature_mean is not None and feature_std is not None:
        safe_std = torch.where(feature_std < 1e-6, torch.ones_like(feature_std), feature_std)
        features = (features - feature_mean) / safe_std
    theoretical_cycles = torch.tensor(
        frame[theoretical_cycle_feature].to_numpy(), dtype=torch.float32
    )
    clock_mhz = torch.tensor(
        frame[get_feature_column_name("gpu_clock_mhz")].to_numpy(), dtype=torch.float32
    )
    predictions = predict_latencies(
        model,
        features,
        theoretical_cycles=theoretical_cycles,
        clock_mhz=clock_mhz,
        target_kind=target_kind,
        launch_overhead_us=launch_overhead_us,
    ).cpu().numpy()
    targets = frame["latency_us"].to_numpy(dtype=np.float32)
    return {
        "mape": mape(targets, predictions),
        "rmse": rmse(targets, predictions),
        "p50_relative_error": percentile_relative_error(targets, predictions, 50),
        "p90_relative_error": percentile_relative_error(targets, predictions, 90),
    }
