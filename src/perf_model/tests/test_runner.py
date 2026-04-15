from __future__ import annotations

import subprocess
from pathlib import Path

from perf_model.profiling.runner import (
    build_cublaslt_bench_compile_cmd,
    get_cublaslt_bench_binary_path,
    parse_cublaslt_bench_payload,
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


def test_parse_cublaslt_bench_payload_includes_kernel_name() -> None:
    payload = parse_cublaslt_bench_payload(
        '{"latency_us": 5.1, "iterations": 20, "device": 4, "gpu_name": "RTX 4090", "kernel_name": "ampere_h16816gemm_128x128_ldg8"}'
    )

    assert payload["kernel_name"] == "ampere_h16816gemm_128x128_ldg8"
    assert payload["latency_us"] == 5.1


def test_run_cublaslt_gemm_bench_parses_kernel_name_from_subprocess_output(monkeypatch) -> None:
    from perf_model.profiling import runner

    def fake_build(repo_root: Path) -> Path:
        return repo_root / ".cache" / "cublaslt_gemm_bench"

    def fake_run(command, check, cwd, capture_output, text):  # noqa: ANN001
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"latency_us": 5.1, "iterations": 20, "device": 4, "gpu_name": "RTX 4090", "kernel_name": "ampere_h16816gemm_128x128_ldg8"}',
            stderr="",
        )

    monkeypatch.setattr(runner, "build_cublaslt_gemm_bench", fake_build)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.run_cublaslt_gemm_bench(
        Path(__file__).resolve().parents[3],
        device=4,
        m=128,
        n=128,
        k=128,
    )

    assert result.kernel_name == "ampere_h16816gemm_128x128_ldg8"
    assert result.device == 4
