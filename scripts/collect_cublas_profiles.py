#!/usr/bin/env python3
"""Collect raw cuBLASLt GEMM profile rows into a CSV."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perf_model.profiling.cublas_profile import (
    extract_main_kernel_from_nsys_cuda_gpu_trace,
    normalize_cublas_profile_row,
)
from perf_model.profiling.runner import build_cublaslt_gemm_bench, get_cublaslt_bench_binary_path, run_cublaslt_gemm_bench


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


def collect_cublas_profile_rows(
    *,
    cases: list[dict[str, int]],
    device: int,
    max_cases: int | None,
    dry_run: bool = False,
) -> list[dict[str, object]]:
    selected = cases[:max_cases] if max_cases is not None else cases
    rows: list[dict[str, object]] = []
    for case in selected:
        if dry_run:
            payload = {
                "latency_us": 1.0,
                "device": device,
                "gpu_name": "RTX 4090",
                "kernel_name": "ampere_h16816gemm_128x128_ldg8_stages_32x1_nn",
            }
        else:
            bench = run_cublaslt_gemm_bench(
                PROJECT_ROOT,
                device=device,
                m=case["M"],
                n=case["N"],
                k=case["K"],
            )
            binary = build_cublaslt_gemm_bench(PROJECT_ROOT)
            with tempfile.TemporaryDirectory(prefix="cublas_nsys_") as tmp_dir:
                report_prefix = Path(tmp_dir) / "report"
                subprocess.run(
                    [
                        "/usr/local/cuda/bin/nsys",
                        "profile",
                        "--trace=cuda",
                        "--sample=none",
                        "--force-overwrite",
                        "true",
                        "--output",
                        str(report_prefix),
                        str(binary),
                        "--device",
                        str(device),
                        "--m",
                        str(case["M"]),
                        "--n",
                        str(case["N"]),
                        "--k",
                        str(case["K"]),
                        "--dtype",
                        "f16",
                        "--iterations",
                        "1",
                        "--warmup",
                        "0",
                    ],
                    check=True,
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                )
                stats_prefix = Path(tmp_dir) / "cuda_gpu_trace"
                subprocess.run(
                    [
                        "/usr/local/cuda/bin/nsys",
                        "stats",
                        "--report",
                        "cuda_gpu_trace",
                        "--format",
                        "csv",
                        "--output",
                        str(stats_prefix),
                        str(report_prefix) + ".nsys-rep",
                    ],
                    check=True,
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                )
                kernel = extract_main_kernel_from_nsys_cuda_gpu_trace(
                    Path(f"{stats_prefix}_cuda_gpu_trace.csv")
                )
            payload = {
                "latency_us": bench.latency_us,
                "device": bench.device,
                "gpu_name": bench.gpu_name,
                "kernel_name": str(kernel["kernel_name"]),
            }
        rows.append(normalize_cublas_profile_row(case, payload))
    return rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cases = iter_cublas_profile_cases()
    rows = collect_cublas_profile_rows(
        cases=cases,
        device=args.device,
        max_cases=args.max_cases,
        dry_run=False,
    )
    write_profile_rows(Path(args.output), rows)


if __name__ == "__main__":
    main()
