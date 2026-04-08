"""Training loop helpers."""

from __future__ import annotations

import torch


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(features)
    loss = loss_fn(predictions, targets)
    loss.backward()
    optimizer.step()
    return float(loss.item())
