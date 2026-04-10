"""Simple MLP baseline for latency regression."""

from __future__ import annotations

import torch
from torch import nn


class LatencyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int] | None = None,
        *,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        sizes = hidden_sizes or [256, 256, 128]
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in sizes:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)
