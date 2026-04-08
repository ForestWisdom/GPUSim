"""Evaluation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from perf_model.model.metrics import mape, percentile_relative_error, rmse
from perf_model.model.predict import predict_latencies


def evaluate_frame(model: torch.nn.Module, frame: pd.DataFrame) -> dict[str, float]:
    feature_columns = [column for column in frame.columns if column.startswith("f_")]
    features = torch.tensor(frame[feature_columns].to_numpy(), dtype=torch.float32)
    predictions = predict_latencies(model, features).cpu().numpy()
    targets = frame["latency_us"].to_numpy(dtype=np.float32)
    return {
        "mape": mape(targets, predictions),
        "rmse": rmse(targets, predictions),
        "p50_relative_error": percentile_relative_error(targets, predictions, 50),
        "p90_relative_error": percentile_relative_error(targets, predictions, 90),
    }
