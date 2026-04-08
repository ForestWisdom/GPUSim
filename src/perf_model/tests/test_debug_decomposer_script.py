from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_debug_decomposer_script_runs_case_a() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "debug_decomposer.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--m",
            "256",
            "--n",
            "256",
            "--k",
            "64",
            "--tb-m",
            "128",
            "--tb-n",
            "128",
            "--tb-k",
            "32",
            "--split-k",
            "1",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "task_count=4" in result.stdout
    assert "logical_output_tiles=4" in result.stdout
    assert "empty_tasks=0" in result.stdout


def test_debug_decomposer_script_shows_cutlass_k_partition_metadata() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "debug_decomposer.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--m",
            "130",
            "--n",
            "129",
            "--k",
            "33",
            "--tb-m",
            "128",
            "--tb-n",
            "128",
            "--tb-k",
            "32",
            "--split-k",
            "2",
            "--dtype",
            "f16",
            "--hide-tasks",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "k_align=8" in result.stdout
    assert "gemm_k_size=24" in result.stdout
    assert "effective_split_k=2" in result.stdout
