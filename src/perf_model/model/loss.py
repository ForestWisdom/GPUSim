"""Loss helpers."""
from __future__ import annotations

import torch
from torch import nn


class MAPELoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(target.abs(), min=self.eps)
        return ((pred - target).abs() / denom).mean()


def build_loss(name: str = "mse") -> nn.Module:
    name = name.lower()
    if name == "mape":
        return MAPELoss()
    if name == "l1":
        return nn.L1Loss()
    return nn.MSELoss()