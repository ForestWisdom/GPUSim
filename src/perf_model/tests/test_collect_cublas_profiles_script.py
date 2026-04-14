from __future__ import annotations

from pathlib import Path

from scripts.collect_cublas_profiles import (
    build_cublas_collect_command,
    collect_cublas_profile_rows,
    iter_cublas_profile_cases,
    write_profile_rows,
)


def test_iter_cublas_profile_cases_contains_regular_and_edge_shapes() -> None:
    cases = list(iter_cublas_profile_cases())

    assert {"M": 128, "N": 128, "K": 128} in cases
    assert {"M": 130, "N": 129, "K": 33} in cases
    assert any(case["K"] >= 1024 for case in cases)


def test_write_profile_rows_emits_csv(tmp_path: Path) -> None:
    output = tmp_path / "profiles.csv"
    rows = [
        {
            "M": 128,
            "N": 128,
            "K": 128,
            "dtype": "f16",
            "device": 4,
            "gpu_name": "RTX 4090",
            "latency_us": 12.5,
            "kernel_name": "main",
            "kernel_index": 0,
            "grid_x": 1,
            "grid_y": 1,
            "grid_z": 1,
            "block_x": 256,
            "block_y": 1,
            "block_z": 1,
            "is_reduction_kernel": False,
            "gemm_call_id": "call-0",
        }
    ]

    write_profile_rows(output, rows)

    text = output.read_text()
    assert "kernel_name" in text
    assert "call-0" in text


def test_build_cublas_collect_command_targets_device_4() -> None:
    cmd = build_cublas_collect_command({"M": 128, "N": 128, "K": 128}, device=4)

    assert "--device 4" in cmd
    assert "--m 128" in cmd
    assert "--n 128" in cmd
    assert "--k 128" in cmd


def test_collect_cublas_profile_rows_returns_kernel_name_rows() -> None:
    rows = collect_cublas_profile_rows(
        cases=[{"M": 128, "N": 128, "K": 128}],
        device=4,
        max_cases=1,
        dry_run=True,
    )

    assert len(rows) == 1
    assert "kernel_name" in rows[0]
