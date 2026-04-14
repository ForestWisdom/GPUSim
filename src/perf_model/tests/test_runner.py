from __future__ import annotations

from pathlib import Path

from perf_model.profiling.runner import (
    build_cublaslt_bench_compile_cmd,
    get_cublaslt_bench_binary_path,
    resolve_cutlass_root,
)


def test_resolve_cutlass_root_falls_back_to_git_common_dir() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    resolved = resolve_cutlass_root(repo_root)

    assert (resolved / "include" / "cutlass" / "cutlass.h").exists()


def test_get_cublaslt_bench_binary_path_points_to_cache() -> None:
    path = get_cublaslt_bench_binary_path()

    assert path.name == "cublaslt_gemm_bench"
    assert ".cache" in str(path)


def test_cublaslt_bench_compile_command_links_cublaslt() -> None:
    cmd = build_cublaslt_bench_compile_cmd()

    assert "tools/cublaslt_gemm_bench.cu" in cmd
    assert "-lcublasLt" in cmd
    assert "-lcublas" in cmd
