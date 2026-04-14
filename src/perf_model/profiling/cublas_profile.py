"""Normalization helpers for cuBLASLt benchmark and profiler output."""

from __future__ import annotations

import json


def is_reduction_kernel_name(kernel_name: str) -> bool:
    lowered = kernel_name.lower()
    return "reduce" in lowered or "splitk" in lowered


def normalize_bench_result(
    problem: dict[str, int],
    bench_result: dict[str, int | float | str],
    kernel_record: dict[str, int | str],
    gemm_call_id: str,
) -> dict[str, int | float | str | bool]:
    kernel_name = str(kernel_record["kernel_name"])
    return {
        "M": int(problem["M"]),
        "N": int(problem["N"]),
        "K": int(problem["K"]),
        "dtype": "f16",
        "device": int(bench_result["device"]),
        "gpu_name": str(bench_result["gpu_name"]),
        "latency_us": float(bench_result["latency_us"]),
        "kernel_name": kernel_name,
        "kernel_index": int(kernel_record["kernel_index"]),
        "grid_x": int(kernel_record["grid_x"]),
        "grid_y": int(kernel_record["grid_y"]),
        "grid_z": int(kernel_record["grid_z"]),
        "block_x": int(kernel_record["block_x"]),
        "block_y": int(kernel_record["block_y"]),
        "block_z": int(kernel_record["block_z"]),
        "is_reduction_kernel": is_reduction_kernel_name(kernel_name),
        "gemm_call_id": gemm_call_id,
    }


def parse_cublas_bench_stdout(stdout: str) -> dict[str, int | float | str]:
    payload = json.loads(stdout)
    return {
        "latency_us": float(payload["latency_us"]),
        "device": int(payload["device"]),
        "gpu_name": str(payload["gpu_name"]),
        "kernel_name": str(payload["kernel_name"]),
    }


def normalize_cublas_profile_row(
    problem: dict[str, int],
    bench_payload: dict[str, int | float | str],
) -> dict[str, int | float | str]:
    return {
        "M": int(problem["M"]),
        "N": int(problem["N"]),
        "K": int(problem["K"]),
        "dtype": "f16",
        "device": int(bench_payload["device"]),
        "gpu_name": str(bench_payload["gpu_name"]),
        "latency_us": float(bench_payload["latency_us"]),
        "kernel_name": str(bench_payload["kernel_name"]),
    }
