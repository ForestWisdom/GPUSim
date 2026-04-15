from __future__ import annotations

from pathlib import Path
import json

from scripts.collect_cublas_profiles import (
    append_profile_rows,
    build_cublas_collect_command,
    collect_cublas_profile_rows,
    iter_cublas_profile_cases,
    parse_args,
    load_collection_state,
    run_batched_collection,
    save_collection_state,
    write_profile_rows,
)


def test_iter_cublas_profile_cases_wide_profile_covers_range_and_target() -> None:
    cases = list(iter_cublas_profile_cases(profile="wide_v1", target_cases=512))

    assert len(cases) == 512
    assert {"M": 16, "N": 128, "K": 32} in cases
    assert {"M": 64, "N": 64, "K": 64} in cases
    assert {"M": 128, "N": 128, "K": 128} in cases
    assert {"M": 130, "N": 129, "K": 33} in cases
    assert {"M": 512, "N": 512, "K": 4096} in cases
    assert max(case["M"] for case in cases) >= 4096
    assert max(case["N"] for case in cases) >= 8192
    assert max(case["K"] for case in cases) >= 4096
    assert len({(case["M"], case["N"], case["K"]) for case in cases}) == len(cases)


def test_iter_cublas_profile_cases_smoke_profile_stays_small() -> None:
    cases = list(iter_cublas_profile_cases(profile="smoke"))

    assert len(cases) == 6
    assert {"M": 130, "N": 129, "K": 33} in cases
    assert {"M": 512, "N": 512, "K": 4096} in cases


def test_parse_args_supports_profile_and_target_cases() -> None:
    args = parse_args(
        [
            "--output",
            "profiles.csv",
            "--profile",
            "wide_v1",
            "--target-cases",
            "2048",
        ]
    )

    assert args.profile == "wide_v1"
    assert args.target_cases == 2048


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


def test_append_profile_rows_preserves_single_header(tmp_path: Path) -> None:
    output = tmp_path / "profiles.csv"
    row = {
        "M": 128,
        "N": 128,
        "K": 128,
        "dtype": "f16",
        "device": 4,
        "gpu_name": "RTX 4090",
        "latency_us": 12.5,
        "kernel_name": "main",
    }

    append_profile_rows(output, [row])
    append_profile_rows(output, [row])

    lines = output.read_text().splitlines()
    assert lines[0].startswith("M,N,K")
    assert len(lines) == 3


def test_collection_state_round_trip(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    payload = {
        "profile_name": "cublas_auto",
        "device": 4,
        "next_case_index": 8,
        "completed_batches": 2,
        "completed_rows": 8,
    }

    save_collection_state(state_path, payload)
    loaded = load_collection_state(state_path)

    assert loaded == payload


def test_run_batched_collection_updates_state_and_writes_batches(tmp_path: Path) -> None:
    output = tmp_path / "profiles.csv"
    state_path = tmp_path / "state.json"
    batches_dir = tmp_path / "batches"
    cases = [
        {"M": 128, "N": 128, "K": 128},
        {"M": 128, "N": 128, "K": 256},
        {"M": 128, "N": 256, "K": 128},
    ]

    state = run_batched_collection(
        cases=cases,
        device=4,
        output=output,
        state_path=state_path,
        batches_dir=batches_dir,
        batch_size=2,
        max_total_cases=3,
        profile_name="cublas_auto",
        dry_run=True,
    )

    assert state["completed_batches"] == 2
    assert state["completed_rows"] == 3
    assert state["next_case_index"] == 3
    assert output.exists()
    assert (batches_dir / "batch_0000.csv").exists()
    assert (batches_dir / "batch_0001.csv").exists()
