from pathlib import Path

import pandas as pd

from scripts.build_cublas_dataset import build_cublas_dataset_frame


def test_build_cublas_dataset_frame_reuses_feature_pipeline(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "M": 128,
                "N": 128,
                "K": 128,
                "dtype": "f16",
                "device": 4,
                "gpu_name": "RTX 4090",
                "latency_us": 5.0,
                "kernel_name": "ampere_h16816gemm_128x128_ldg8_stages_32x1_nn",
            }
        ]
    )

    dataset = build_cublas_dataset_frame(frame, gpu_yaml="configs/gpu/4090.yaml")

    assert "latency_us" in dataset.columns
    assert any(column.startswith("f_") for column in dataset.columns)
