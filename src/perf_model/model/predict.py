"""Prediction helpers."""

from __future__ import annotations

import torch


def predict_latencies(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(features)
