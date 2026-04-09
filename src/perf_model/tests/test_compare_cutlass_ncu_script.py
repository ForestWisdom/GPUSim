from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_compare_cutlass_ncu_script_help() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "compare_cutlass_ncu.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--ncu-csv" in result.stdout
    assert "--gpu-config" in result.stdout
    assert "--total-metric" in result.stdout
    assert "--metric-scale" in result.stdout
    assert "--max-metric" in result.stdout
    assert "--max-metric-scale" in result.stdout
    assert "--iterations" in result.stdout
    assert "--warmup" in result.stdout


def test_compare_cutlass_ncu_script_compares_model_to_csv(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "compare_cutlass_ncu.py"
    csv_path = tmp_path / "ncu.csv"
    csv_path.write_text(
        "\n".join(
            [
                '"ID","Kernel Name","sm__inst_executed_pipe_tensor_op_hmma_v2.sum","sm__inst_executed_pipe_tensor_op_hmma_v2.max"',
                '"","","inst","inst"',
                '"0","cutlass_kernel","4194304","1048576"',
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--gpu-config",
            "configs/gpu/4090.yaml",
            "--m",
            "128",
            "--n",
            "128",
            "--k",
            "128",
            "--tb-m",
            "128",
            "--tb-n",
            "128",
            "--tb-k",
            "32",
            "--skip-probe",
            "--ncu-csv",
            str(csv_path),
            "--metric-scale",
            "1.0",
            "--max-metric-scale",
            "1.0",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["model"]["task_count"] == 1
    assert payload["model"]["total_tensor_ops"] == 4194304.0
    assert payload["ncu"]["total_tensor_ops"] == 4194304.0
    assert payload["ncu"]["max_sm_tensor_ops"] == 1048576.0
