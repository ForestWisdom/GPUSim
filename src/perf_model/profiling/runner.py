"""Helpers for building and running local profiling binaries."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ProfileResult:
    latency_us: float
    repeats: int


@dataclass(slots=True)
class CutlassBenchmarkResult(ProfileResult):
    task_count: int
    swizzle_log_tile: int
    grid_tiled_shape: tuple[int, int, int]


def resolve_cutlass_root(repo_root: Path) -> Path:
    env_root = os.environ.get("CUTLASS_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(repo_root / "thirdparty" / "cutlass")

    try:
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=repo_root,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        common_dir = ""
    if common_dir:
        candidates.append(Path(common_dir).resolve().parent / "thirdparty" / "cutlass")

    for candidate in candidates:
        if (candidate / "include" / "cutlass" / "cutlass.h").exists():
            return candidate

    raise FileNotFoundError("could not find CUTLASS headers; set CUTLASS_ROOT or add thirdparty/cutlass")


def build_cutlass_gemm_bench(repo_root: Path) -> Path:
    source = repo_root / "tools" / "cutlass_gemm_bench.cu"
    include_root = resolve_cutlass_root(repo_root)
    binary = repo_root / ".cache" / "cutlass_gemm_bench"
    binary.parent.mkdir(exist_ok=True)

    if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        return binary

    command = [
        "nvcc",
        "-std=c++17",
        "-O2",
        "--expt-relaxed-constexpr",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_75,code=compute_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_89,code=compute_89",
        f"-I{include_root / 'include'}",
        f"-I{include_root / 'tools' / 'util' / 'include'}",
        str(source),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, cwd=repo_root)
    return binary


def run_cutlass_gemm_bench(
    repo_root: Path,
    *,
    device: int,
    m: int,
    n: int,
    k: int,
    tb_m: int,
    tb_n: int,
    tb_k: int,
    split_k: int,
    swizzle: str,
    iterations: int,
    warmup: int,
) -> CutlassBenchmarkResult:
    binary = build_cutlass_gemm_bench(repo_root)
    command = [
        str(binary),
        "--device",
        str(device),
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--tb-m",
        str(tb_m),
        "--tb-n",
        str(tb_n),
        "--tb-k",
        str(tb_k),
        "--split-k",
        str(split_k),
        "--swizzle",
        swizzle,
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
    ]
    result = subprocess.run(command, check=True, cwd=repo_root, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    return CutlassBenchmarkResult(
        latency_us=float(payload["latency_us"]),
        repeats=int(payload["iterations"]),
        task_count=int(payload["task_count"]),
        swizzle_log_tile=int(payload["swizzle_log_tile"]),
        grid_tiled_shape=tuple(int(item) for item in payload["grid_tiled_shape"]),
    )
