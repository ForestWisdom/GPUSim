import pandas as pd
import torch

from perf_model.model.mlp import LatencyMLP
from perf_model.pipelines.eval_pipeline import evaluate_frame
from perf_model.pipelines.train_pipeline import train_from_frame


def test_train_from_frame_returns_normalization_metadata() -> None:
    frame = pd.DataFrame(
        {
            "latency_us": [10.0, 12.0, 14.0, 16.0],
            "f_0": [1.0, 2.0, 3.0, 4.0],
            "f_1": [100.0, 110.0, 120.0, 130.0],
        }
    )

    result = train_from_frame(frame, hidden_sizes=[8], epochs=3, patience=2)

    assert result.feature_columns == ["f_0", "f_1"]
    assert tuple(result.feature_mean.shape) == (2,)
    assert tuple(result.feature_std.shape) == (2,)
    assert result.hidden_sizes == [8]


def test_evaluate_frame_uses_training_normalization_stats() -> None:
    frame = pd.DataFrame(
        {
            "latency_us": [0.0, 0.0],
            "f_0": [3.0, 5.0],
            "f_1": [12.0, 16.0],
        }
    )
    mean = torch.tensor([1.0, 2.0], dtype=torch.float32)
    std = torch.tensor([2.0, 2.0], dtype=torch.float32)

    class SummingModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs.sum(dim=1)

    metrics = evaluate_frame(SummingModel(), frame, feature_mean=mean, feature_std=std)

    assert metrics["rmse"] > 0.0
    expected_predictions = torch.tensor([6.0, 9.0], dtype=torch.float32)
    assert metrics["rmse"] == torch.sqrt((expected_predictions.pow(2).mean())).item()


def test_checkpoint_payload_round_trip_reconstructs_model_inputs() -> None:
    model = LatencyMLP(input_dim=3, hidden_sizes=[4, 2])
    payload = {
        "model_state_dict": model.state_dict(),
        "feature_columns": ["f_0", "f_1", "f_2"],
        "feature_mean": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        "feature_std": torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32),
        "hidden_sizes": [4, 2],
    }

    restored = LatencyMLP(input_dim=len(payload["feature_columns"]), hidden_sizes=payload["hidden_sizes"])
    restored.load_state_dict(payload["model_state_dict"])

    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
