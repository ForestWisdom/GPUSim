from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_validate_cutlass_decomposer_script_help() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "validate_cutlass_decomposer.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--swizzle" in result.stdout
    assert "--split-k" in result.stdout
