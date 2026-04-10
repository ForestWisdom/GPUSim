import pandas as pd
import torch

from perf_model.features.feature_vector import FEATURE_VECTOR_FIELDS
from perf_model.model.mlp import LatencyMLP
from perf_model.pipelines.eval_pipeline import evaluate_frame
from perf_model.pipelines.train_pipeline import train_from_frame


def test_train_from_frame_returns_normalization_metadata() -> None:
    feature_count = len(FEATURE_VECTOR_FIELDS)
    frame = pd.DataFrame(
        {"latency_us": [10.0, 12.0, 14.0, 16.0], **{f"f_{idx}": [0.0] * 4 for idx in range(feature_count)}}
    )
    frame["f_5"] = [1000.0] * 4
    frame["f_33"] = [5000.0, 6000.0, 7000.0, 8000.0]

    result = train_from_frame(frame, hidden_sizes=[8], epochs=3, patience=2)

    assert result.feature_columns[0] == "f_0"
    assert result.feature_columns[-1] == f"f_{feature_count - 1}"
    assert tuple(result.feature_mean.shape) == (feature_count,)
    assert tuple(result.feature_std.shape) == (feature_count,)
    assert result.hidden_sizes == [8]
    assert result.best_epoch >= 0
    assert set(result.val_metrics) == {"mape", "rmse"}
    assert set(result.train_metrics) == {"mape", "rmse"}
    assert result.val_metrics["rmse"] >= 0.0
    assert result.val_metrics["mape"] >= 0.0
    assert result.target_kind == "efficiency"
    assert result.theoretical_cycle_feature == "f_33"
    assert result.loss_name == "mape"
    assert result.dropout == 0.1
    assert result.use_batch_norm is True


def test_evaluate_frame_reconstructs_latency_from_efficiency_prediction() -> None:
    frame = pd.DataFrame(
        {
            "latency_us": [10.0, 10.0],
            **{f"f_{idx}": [0.0, 0.0] for idx in range(len(FEATURE_VECTOR_FIELDS))},
        }
    )
    frame["f_5"] = [1000.0, 1000.0]
    frame["f_33"] = [5000.0, 5000.0]
    mean = torch.zeros(len(FEATURE_VECTOR_FIELDS), dtype=torch.float32)
    std = torch.ones(len(FEATURE_VECTOR_FIELDS), dtype=torch.float32)

    class ConstantEfficiencyModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.full((inputs.shape[0],), 0.5, dtype=inputs.dtype)

    metrics = evaluate_frame(
        ConstantEfficiencyModel(),
        frame,
        feature_mean=mean,
        feature_std=std,
        target_kind="efficiency",
        theoretical_cycle_feature="f_33",
    )

    assert metrics["rmse"] == 0.0
    assert metrics["mape"] == 0.0


def test_checkpoint_payload_round_trip_reconstructs_model_inputs() -> None:
    model = LatencyMLP(input_dim=3, hidden_sizes=[4, 2])
    payload = {
        "model_state_dict": model.state_dict(),
        "feature_columns": ["f_0", "f_1", "f_2"],
        "feature_mean": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        "feature_std": torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32),
        "hidden_sizes": [4, 2],
        "best_epoch": 3,
        "best_val_loss": 1.25,
        "train_metrics": {"mape": 8.0, "rmse": 2.0},
        "val_metrics": {"mape": 10.0, "rmse": 3.0},
        "target_kind": "efficiency",
        "theoretical_cycle_feature": "f_33",
        "loss_name": "mape",
        "dropout": 0.1,
        "use_batch_norm": True,
    }

    restored = LatencyMLP(
        input_dim=len(payload["feature_columns"]),
        hidden_sizes=payload["hidden_sizes"],
        dropout=payload["dropout"],
        use_batch_norm=payload["use_batch_norm"],
    )
    restored.load_state_dict(payload["model_state_dict"])

    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    assert payload["best_epoch"] == 3
    assert payload["val_metrics"]["mape"] == 10.0
    assert payload["target_kind"] == "efficiency"
    assert payload["theoretical_cycle_feature"] == "f_33"
    assert payload["loss_name"] == "mape"
    assert payload["dropout"] == 0.1
    assert payload["use_batch_norm"] is True
