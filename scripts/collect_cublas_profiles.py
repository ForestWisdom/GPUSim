#!/usr/bin/env python3
"""Collect raw cuBLASLt GEMM profile rows into a CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def _dedupe_cases(cases: list[dict[str, int]]) -> list[dict[str, int]]:
    deduped: list[dict[str, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for row in cases:
        key = (row["M"], row["N"], row["K"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _log2_grid(start: int, stop: int, points_per_octave: int) -> list[int]:
    values: set[int] = set()
    current = start
    while current <= stop:
        values.add(int(current))
        current *= 2
    octaves = max(1, int(math.log2(stop / start)))
    total_points = max(points_per_octave * octaves, 1)
    for index in range(total_points + 1):
        ratio = index / total_points
        value = int(round(start * ((stop / start) ** ratio)))
        values.add(max(start, min(stop, value)))
    return sorted(values)


def _candidate_values(axis: str) -> list[int]:
    if axis == "M":
        dense_small = [16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]
        medium = [320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048]
        large = [2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192]
        return sorted(set(dense_small + medium + large))
    if axis == "N":
        base = _log2_grid(128, 16384, points_per_octave=5)
        extras = [160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1536, 2048, 3072, 4096]
        return sorted(set(base + extras))
    if axis == "K":
        base = _log2_grid(32, 16384, points_per_octave=6)
        extras = [33, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
        return sorted(set(base + extras))
    raise ValueError(f"unsupported axis: {axis}")


def _edge_cases() -> list[dict[str, int]]:
    return [
        {"M": 16, "N": 128, "K": 32},
        {"M": 32, "N": 128, "K": 33},
        {"M": 64, "N": 64, "K": 64},
        {"M": 96, "N": 160, "K": 48},
        {"M": 127, "N": 255, "K": 95},
        {"M": 128, "N": 128, "K": 128},
        {"M": 129, "N": 128, "K": 128},
        {"M": 130, "N": 129, "K": 33},
        {"M": 191, "N": 257, "K": 65},
        {"M": 255, "N": 255, "K": 255},
        {"M": 256, "N": 256, "K": 128},
        {"M": 257, "N": 255, "K": 65},
        {"M": 384, "N": 128, "K": 3072},
        {"M": 512, "N": 512, "K": 4096},
        {"M": 768, "N": 4096, "K": 1536},
        {"M": 1024, "N": 8192, "K": 2048},
        {"M": 2048, "N": 4096, "K": 4096},
        {"M": 4096, "N": 4096, "K": 4096},
        {"M": 8192, "N": 16384, "K": 8192},
    ]


def iter_cublas_profile_cases(
    *,
    profile: str = "wide_v1",
    target_cases: int = 4096,
) -> list[dict[str, int]]:
    if profile not in {"smoke", "wide_v1"}:
        raise ValueError(f"unsupported profile: {profile}")

    if profile == "smoke":
        return _dedupe_cases(
            [
                {"M": 64, "N": 64, "K": 64},
                {"M": 128, "N": 128, "K": 128},
                {"M": 256, "N": 256, "K": 128},
                {"M": 128, "N": 256, "K": 1024},
                {"M": 130, "N": 129, "K": 33},
                {"M": 512, "N": 512, "K": 4096},
            ]
        )

    m_values = _candidate_values("M")
    n_values = _candidate_values("N")
    k_values = _candidate_values("K")

    cases = list(_edge_cases())
    seen = {(row["M"], row["N"], row["K"]) for row in cases}
    pools: dict[str, list[dict[str, int]]] = {
        "small_m": [],
        "medium_m": [],
        "large_m": [],
    }

    for mi, m in enumerate(m_values):
        for ni, n in enumerate(n_values):
            for ki, k in enumerate(k_values):
                score = (mi * 17 + ni * 31 + ki * 47 + (m // 16) + (n // 64) + (k // 32)) % 11
                keep = False
                bucket = "small_m"
                if m <= 256:
                    keep = n <= 4096 or score in {0, 3, 7}
                    bucket = "small_m"
                elif m <= 2048:
                    keep = (
                        (n <= 4096 and k <= 4096 and score in {1, 4, 8})
                        or (n > 4096 and score in {2, 5})
                        or (k > 4096 and score in {6, 9})
                    )
                    bucket = "medium_m"
                else:
                    keep = score in {0, 2, 5, 8}
                    bucket = "large_m"
                if not keep:
                    continue
                key = (m, n, k)
                if key in seen:
                    continue
                pools[bucket].append({"M": m, "N": n, "K": k})

    remaining = max(target_cases - len(cases), 0)
    quota_small = int(remaining * 0.45)
    quota_medium = int(remaining * 0.35)
    quota_large = remaining - quota_small - quota_medium
    quotas = {
        "small_m": quota_small,
        "medium_m": quota_medium,
        "large_m": quota_large,
    }

    for bucket_name in ("small_m", "medium_m", "large_m"):
        for row in pools[bucket_name]:
            if quotas[bucket_name] <= 0 or len(cases) >= target_cases:
                break
            key = (row["M"], row["N"], row["K"])
            if key in seen:
                continue
            seen.add(key)
            cases.append(row)
            quotas[bucket_name] -= 1

    for bucket_name in ("small_m", "medium_m", "large_m"):
        if len(cases) >= target_cases:
            break
        for row in pools[bucket_name]:
            if len(cases) >= target_cases:
                break
            key = (row["M"], row["N"], row["K"])
            if key in seen:
                continue
            seen.add(key)
            cases.append(row)

    return _dedupe_cases(cases)


def write_profile_rows(output: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_profile_rows(output: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
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
    parser.add_argument("--profile", default="wide_v1")
    parser.add_argument("--target-cases", type=int, default=4096)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-total-cases", type=int, default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--profile-name", default="cublas_auto")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--batches-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_collection_state(path: Path) -> dict[str, int | str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid collection state in {path}")
    return payload


def save_collection_state(path: Path, payload: dict[str, int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def collect_cublas_profile_batch(
    *,
    cases: list[dict[str, int]],
    device: int,
    start_index: int,
    batch_size: int,
    dry_run: bool = False,
) -> tuple[list[dict[str, object]], int]:
    selected = cases[start_index : start_index + batch_size]
    rows = collect_cublas_profile_rows(
        cases=selected,
        device=device,
        max_cases=None,
        dry_run=dry_run,
    )
    return rows, len(selected)


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


def run_batched_collection(
    *,
    cases: list[dict[str, int]],
    device: int,
    output: Path,
    state_path: Path,
    batches_dir: Path,
    batch_size: int,
    max_total_cases: int | None,
    profile_name: str,
    dry_run: bool = False,
) -> dict[str, int | str]:
    state = load_collection_state(state_path)
    next_case_index = int(state.get("next_case_index", 0))
    completed_batches = int(state.get("completed_batches", 0))
    completed_rows = int(state.get("completed_rows", 0))

    limit = len(cases) if max_total_cases is None else min(len(cases), max_total_cases)
    output.parent.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)

    while next_case_index < limit:
        current_batch_size = min(batch_size, limit - next_case_index)
        rows, consumed = collect_cublas_profile_batch(
            cases=cases,
            device=device,
            start_index=next_case_index,
            batch_size=current_batch_size,
            dry_run=dry_run,
        )
        batch_path = batches_dir / f"batch_{completed_batches:04d}.csv"
        write_profile_rows(batch_path, rows)
        append_profile_rows(output, rows)

        next_case_index += consumed
        completed_batches += 1
        completed_rows += len(rows)
        state = {
            "profile_name": profile_name,
            "device": device,
            "next_case_index": next_case_index,
            "completed_batches": completed_batches,
            "completed_rows": completed_rows,
        }
        save_collection_state(state_path, state)

    return state


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cases = iter_cublas_profile_cases(profile=args.profile, target_cases=args.target_cases)
    if args.batch_size is None:
        rows = collect_cublas_profile_rows(
            cases=cases,
            device=args.device,
            max_cases=args.max_cases,
            dry_run=args.dry_run,
        )
        output = Path(args.output)
        if args.append:
            append_profile_rows(output, rows)
        else:
            write_profile_rows(output, rows)
        print(f"rows={len(rows)}")
        print(f"output={output}")
        return

    output = Path(args.output)
    state_path = Path(args.state_path) if args.state_path else output.with_suffix(".state.json")
    batches_dir = Path(args.batches_dir) if args.batches_dir else output.parent / f"{output.stem}_batches"
    state = run_batched_collection(
        cases=cases,
        device=args.device,
        output=output,
        state_path=state_path,
        batches_dir=batches_dir,
        batch_size=args.batch_size,
        max_total_cases=args.max_total_cases if args.max_total_cases is not None else args.max_cases,
        profile_name=args.profile_name,
        dry_run=args.dry_run,
    )
    print(f"completed_rows={state['completed_rows']}")
    print(f"completed_batches={state['completed_batches']}")
    print(f"next_case_index={state['next_case_index']}")
    print(f"output={output}")
    print(f"state={state_path}")


if __name__ == "__main__":
    main()
