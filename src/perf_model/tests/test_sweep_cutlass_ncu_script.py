from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_sweep_cutlass_ncu_script_help() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "sweep_cutlass_ncu.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--gpu-config" in result.stdout
    assert "--total-metric" in result.stdout
    assert "--max-metric" in result.stdout
    assert "--ncu-prefix" in result.stdout
    assert "--output-json" in result.stdout
