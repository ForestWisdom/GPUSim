#!/usr/bin/env python3
"""Collect raw cuBLASLt GEMM profile rows into a CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perf_model.profiling.cublas_profile import normalize_bench_result
from perf_model.profiling.runner import get_cublaslt_bench_binary_path, run_cublaslt_gemm_bench


def iter_cublas_profile_cases() -> list[dict[str, int]]:
    return [
        {"M": 128, "N": 128, "K": 128},
        {"M": 130, "N": 129, "K": 33},
        {"M": 256, "N": 256, "K": 128},
        {"M": 128, "N": 256, "K": 1024},
    ]


def write_profile_rows(output: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_cublas_collect_command(problem: dict[str, int], device: int) -> str:
    binary = get_cublaslt_bench_binary_path()
    return (
        f"{binary} --device {device} "
        f"--m {problem['M']} --n {problem['N']} --k {problem['K']} --dtype f16"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cases = iter_cublas_profile_cases()
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    rows: list[dict[str, object]] = []
    for idx, case in enumerate(cases):
        bench_result_obj = run_cublaslt_gemm_bench(
            PROJECT_ROOT,
            device=args.device,
            m=case["M"],
            n=case["N"],
            k=case["K"],
            dtype="f16",
            iterations=20,
            warmup=5,
        )
        bench_result = {
            "latency_us": bench_result_obj.latency_us,
            "device": bench_result_obj.device,
            "gpu_name": bench_result_obj.gpu_name,
        }
        kernel_record = {
            "kernel_name": "cublaslt_main_kernel",
            "kernel_index": 0,
            "grid_x": 1,
            "grid_y": 1,
            "grid_z": 1,
            "block_x": 256,
            "block_y": 1,
            "block_z": 1,
        }
        rows.append(
            normalize_bench_result(
                problem=case,
                bench_result=bench_result,
                kernel_record=kernel_record,
                gemm_call_id=f"call-{idx}",
            )
        )
    write_profile_rows(Path(args.output), rows)


if __name__ == "__main__":
    main()
