from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_dataset import main


def test_build_dataset_cli_keeps_override_columns(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "M": 128,
                "N": 128,
                "K": 64,
                "split_k_slices": 2,
                "swizzle": "Identity2",
                "latency_us": 9.0,
            }
        ]
    )
    raw_path = tmp_path / "raw.csv"
    out_path = tmp_path / "processed.csv"
    raw.to_csv(raw_path, index=False)

    repo_root = Path(__file__).resolve().parents[3]
    main(
        [
            "--input",
            str(raw_path),
            "--output",
            str(out_path),
            "--gpu",
            str(repo_root / "configs" / "gpu" / "4090.yaml"),
            "--kernel",
            str(repo_root / "configs" / "kernels" / "cutlass_gemm_tensorop.yaml"),
        ]
    )

    output = pd.read_csv(out_path)
    assert output["split_k_slices"].tolist() == [2]
    assert output["swizzle"].tolist() == ["Identity2"]
