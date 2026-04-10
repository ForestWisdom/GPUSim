#!/usr/bin/env python3
"""Collect raw CUTLASS GEMM benchmark profiles into a CSV."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perf_model.common.types import KernelMeta
from perf_model.common.utils import load_yaml
from perf_model.kernel_desc.parser import load_kernel_meta
from perf_model.profiling.runner import run_cutlass_gemm_bench


DEFAULT_M_VALUES = [128, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
DEFAULT_N_VALUES = [128, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
DEFAULT_K_VALUES = [32, 33, 64, 65, 96, 127, 128, 129, 255, 256, 384, 512, 1024, 2048, 4096]
DEFAULT_SWIZZLES = ["Identity", "Identity2", "Identity4"]
DEFAULT_SPLIT_K_VALUES = [1, 2]


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None or not value.strip():
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output raw CSV path")
    parser.add_argument("--gpu", required=True, help="GPU yaml config")
    parser.add_argument("--kernel", required=True, help="Kernel yaml config")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m-values", default=None)
    parser.add_argument("--n-values", default=None)
    parser.add_argument("--k-values", default=None)
    parser.add_argument("--split-k-values", default=None)
    parser.add_argument("--swizzles", default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args(argv)


def load_gpu_name(path: str) -> str:
    config = load_yaml(path)
    return str(config["name"])


def iter_profile_cases(
    kernel_meta: KernelMeta,
    *,
    m_values: list[int],
    n_values: list[int],
    k_values: list[int],
    split_k_values: list[int],
    swizzles: list[str],
    max_cases: int | None,
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for m, n, k, split_k, swizzle in product(
        m_values,
        n_values,
        k_values,
        split_k_values,
        swizzles,
    ):
        rows.append(
            {
                "M": m,
                "N": n,
                "K": k,
                "split_k_slices": split_k,
                "swizzle": swizzle,
                "tb_m": kernel_meta.threadblock_shape[0],
                "tb_n": kernel_meta.threadblock_shape[1],
                "tb_k": kernel_meta.threadblock_shape[2],
            }
        )
        if max_cases is not None and len(rows) >= max_cases:
            break
    return rows


def collect_profiles(
    *,
    repo_root: Path,
    gpu_name: str,
    kernel_meta: KernelMeta,
    device: int,
    cases: list[dict[str, int | str]],
    iterations: int,
    warmup: int,
) -> pd.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    for case in cases:
        effective_kernel = replace(
            kernel_meta,
            swizzle=str(case["swizzle"]),
            split_k_default=int(case["split_k_slices"]),
        )
        row: dict[str, int | float | str] = {
            "gpu_name": gpu_name,
            "kernel_name": effective_kernel.name,
            "device": device,
            "dtype": effective_kernel.dtype,
            "M": int(case["M"]),
            "N": int(case["N"]),
            "K": int(case["K"]),
            "split_k_slices": int(case["split_k_slices"]),
            "swizzle": str(case["swizzle"]),
            "tb_m": int(case["tb_m"]),
            "tb_n": int(case["tb_n"]),
            "tb_k": int(case["tb_k"]),
            "iterations": iterations,
            "warmup": warmup,
        }
        try:
            result = run_cutlass_gemm_bench(
                repo_root,
                device=device,
                m=int(case["M"]),
                n=int(case["N"]),
                k=int(case["K"]),
                tb_m=int(case["tb_m"]),
                tb_n=int(case["tb_n"]),
                tb_k=int(case["tb_k"]),
                split_k=int(case["split_k_slices"]),
                swizzle=str(case["swizzle"]),
                iterations=iterations,
                warmup=warmup,
            )
            row.update(
                {
                    "status": "ok",
                    "error": "",
                    "latency_us": result.latency_us,
                    "task_count": result.task_count,
                    "swizzle_log_tile": result.swizzle_log_tile,
                    "grid_m": result.grid_tiled_shape[0],
                    "grid_n": result.grid_tiled_shape[1],
                    "grid_k": result.grid_tiled_shape[2],
                }
            )
        except Exception as exc:  # pragma: no cover
            row.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "latency_us": float("nan"),
                    "task_count": -1,
                    "swizzle_log_tile": -1,
                    "grid_m": -1,
                    "grid_n": -1,
                    "grid_k": -1,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    kernel_meta = load_kernel_meta(args.kernel)
    gpu_name = load_gpu_name(args.gpu)
    cases = iter_profile_cases(
        kernel_meta,
        m_values=parse_int_list(args.m_values, DEFAULT_M_VALUES),
        n_values=parse_int_list(args.n_values, DEFAULT_N_VALUES),
        k_values=parse_int_list(args.k_values, DEFAULT_K_VALUES),
        split_k_values=parse_int_list(args.split_k_values, DEFAULT_SPLIT_K_VALUES),
        swizzles=parse_str_list(args.swizzles, DEFAULT_SWIZZLES),
        max_cases=args.max_cases,
    )
    frame = collect_profiles(
        repo_root=PROJECT_ROOT,
        gpu_name=gpu_name,
        kernel_meta=kernel_meta,
        device=args.device,
        cases=cases,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    ok_count = int((frame["status"] == "ok").sum())
    error_count = int((frame["status"] == "error").sum())
    print(f"wrote_rows={len(frame)}")
    print(f"ok_rows={ok_count}")
    print(f"error_rows={error_count}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
