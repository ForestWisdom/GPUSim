"""Loss helpers."""

from __future__ import annotations

from torch import nn


def build_loss(name: str = "mse") -> nn.Module:
    if name == "mape":
        return nn.L1Loss()
    return nn.MSELoss()
