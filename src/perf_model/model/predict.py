"""Prediction helpers."""

from __future__ import annotations

import torch


def predict_efficiency(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(features)


def reconstruct_latencies(
    predicted_efficiency: torch.Tensor,
    theoretical_cycles: torch.Tensor,
    clock_mhz: torch.Tensor,
    launch_overhead_us: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    safe_efficiency = torch.clamp(predicted_efficiency, min=eps)
    safe_clock = torch.clamp(clock_mhz, min=eps)
    return theoretical_cycles / (safe_efficiency * safe_clock) + launch_overhead_us


def predict_latencies(
    model: torch.nn.Module,
    features: torch.Tensor,
    *,
    theoretical_cycles: torch.Tensor | None = None,
    clock_mhz: torch.Tensor | None = None,
    target_kind: str = "latency",
    launch_overhead_us: float = 0.0,
) -> torch.Tensor:
    predictions = predict_efficiency(model, features)
    if target_kind == "efficiency":
        if theoretical_cycles is None or clock_mhz is None:
            raise ValueError("efficiency prediction requires theoretical_cycles and clock_mhz")
        return reconstruct_latencies(
            predictions,
            theoretical_cycles,
            clock_mhz,
            launch_overhead_us=launch_overhead_us,
        )
    return predictions
