from torch import nn

from perf_model.model.loss import MAPELoss, build_loss
from perf_model.model.mlp import LatencyMLP


def test_build_loss_defaults_to_mape() -> None:
    loss = build_loss()

    assert isinstance(loss, MAPELoss)


def test_latency_mlp_includes_batch_norm_and_dropout_by_default() -> None:
    model = LatencyMLP(input_dim=8, hidden_sizes=[16, 8])

    batch_norm_layers = [layer for layer in model.network if isinstance(layer, nn.BatchNorm1d)]
    dropout_layers = [layer for layer in model.network if isinstance(layer, nn.Dropout)]

    assert len(batch_norm_layers) == 2
    assert len(dropout_layers) == 2
    assert all(layer.p == 0.1 for layer in dropout_layers)
