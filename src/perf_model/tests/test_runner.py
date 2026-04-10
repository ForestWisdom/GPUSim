from __future__ import annotations

from pathlib import Path

from perf_model.profiling.runner import resolve_cutlass_root


def test_resolve_cutlass_root_falls_back_to_git_common_dir() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    resolved = resolve_cutlass_root(repo_root)

    assert (resolved / "include" / "cutlass" / "cutlass.h").exists()
