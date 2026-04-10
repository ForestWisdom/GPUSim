from __future__ import annotations

from pathlib import Path

import pandas as pd

from perf_model.profiling.runner import CutlassBenchmarkResult
from scripts import collect_profiles


def test_collect_profiles_cli_writes_rows_with_runner_results(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_cutlass_gemm_bench(repo_root: Path, **kwargs: object) -> CutlassBenchmarkResult:
        calls.append(kwargs)
        return CutlassBenchmarkResult(
            latency_us=12.5,
            repeats=int(kwargs["iterations"]),
            task_count=4,
            swizzle_log_tile=0,
            grid_tiled_shape=(2, 2, 1),
        )

    monkeypatch.setattr(collect_profiles, "run_cutlass_gemm_bench", fake_run_cutlass_gemm_bench)

    output_path = tmp_path / "profiles.csv"
    repo_root = Path(__file__).resolve().parents[3]
    collect_profiles.main(
        [
            "--output",
            str(output_path),
            "--gpu",
            str(repo_root / "configs" / "gpu" / "4090.yaml"),
            "--kernel",
            str(repo_root / "configs" / "kernels" / "cutlass_gemm_tensorop.yaml"),
            "--m-values",
            "128",
            "--n-values",
            "128,256",
            "--k-values",
            "64",
            "--split-k-values",
            "1",
            "--swizzles",
            "Identity,Identity4",
            "--iterations",
            "3",
            "--warmup",
            "0",
        ]
    )

    frame = pd.read_csv(output_path)

    assert len(frame) == 4
    assert set(frame["status"]) == {"ok"}
    assert frame["latency_us"].tolist() == [12.5, 12.5, 12.5, 12.5]
    assert frame["swizzle"].tolist() == ["Identity", "Identity4", "Identity", "Identity4"]
    assert calls[0]["tb_m"] == 128
    assert calls[0]["tb_n"] == 128
    assert calls[0]["tb_k"] == 32


def test_collect_profiles_cli_records_failures_without_crashing(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_run_cutlass_gemm_bench(repo_root: Path, **kwargs: object) -> CutlassBenchmarkResult:
        raise RuntimeError("bench failed")

    monkeypatch.setattr(collect_profiles, "run_cutlass_gemm_bench", fake_run_cutlass_gemm_bench)

    output_path = tmp_path / "profiles.csv"
    repo_root = Path(__file__).resolve().parents[3]
    collect_profiles.main(
        [
            "--output",
            str(output_path),
            "--gpu",
            str(repo_root / "configs" / "gpu" / "4090.yaml"),
            "--kernel",
            str(repo_root / "configs" / "kernels" / "cutlass_gemm_tensorop.yaml"),
            "--m-values",
            "128",
            "--n-values",
            "128",
            "--k-values",
            "64",
            "--split-k-values",
            "1",
            "--swizzles",
            "Identity",
            "--iterations",
            "3",
            "--warmup",
            "0",
        ]
    )

    frame = pd.read_csv(output_path)

    assert len(frame) == 1
    assert frame.loc[0, "status"] == "error"
    assert "bench failed" in frame.loc[0, "error"]
