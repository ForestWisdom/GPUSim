from __future__ import annotations

from pathlib import Path

from perf_model.validation.ncu_sweep import build_ncu_profile_command, summarize_ncu_sweep_results


def test_build_ncu_profile_command_includes_total_and_max_metrics(tmp_path: Path) -> None:
    command = build_ncu_profile_command(
        ncu_bin="/usr/local/cuda/bin/ncu",
        ncu_prefix=["sudo", "-S"],
        binary=tmp_path / "cutlass_gemm_bench",
        output_csv=tmp_path / "capture.csv",
        device=7,
        m=256,
        n=256,
        k=128,
        tb_m=128,
        tb_n=128,
        tb_k=32,
        split_k=1,
        swizzle="Identity",
        iterations=1,
        warmup=1,
        metric_names=[
            "sm__inst_executed_pipe_tensor_op_hmma_v2.sum",
            "sm__inst_executed_pipe_tensor_op_hmma_v2.max",
        ],
    )

    assert command[:3] == ["sudo", "-S", "/usr/local/cuda/bin/ncu"]
    assert "--metrics" in command
    assert (
        command[command.index("--metrics") + 1]
        == "sm__inst_executed_pipe_tensor_op_hmma_v2.sum,sm__inst_executed_pipe_tensor_op_hmma_v2.max"
    )
    assert "--log-file" in command
    assert str(tmp_path / "capture.csv") in command


def test_summarize_ncu_sweep_results_counts_matches_and_support() -> None:
    summary = summarize_ncu_sweep_results(
        [
            {
                "task_count_match": True,
                "total_tensor_ops_match": True,
                "max_sm_supported": True,
                "max_sm_tensor_ops_match": True,
                "total_tensor_ops_rel_error": 0.0,
                "max_sm_tensor_ops_rel_error": 0.0,
            },
            {
                "task_count_match": False,
                "total_tensor_ops_match": True,
                "max_sm_supported": True,
                "max_sm_tensor_ops_match": False,
                "total_tensor_ops_rel_error": 0.01,
                "max_sm_tensor_ops_rel_error": 0.2,
            },
            {
                "task_count_match": True,
                "total_tensor_ops_match": False,
                "max_sm_supported": False,
                "max_sm_tensor_ops_match": None,
                "total_tensor_ops_rel_error": 0.15,
                "max_sm_tensor_ops_rel_error": None,
            },
        ]
    )

    assert summary["total_cases"] == 3
    assert summary["task_count_matched_cases"] == 2
    assert summary["total_tensor_ops_matched_cases"] == 2
    assert summary["max_sm_supported_cases"] == 2
    assert summary["max_sm_tensor_ops_matched_cases"] == 1
    assert summary["fully_matched_cases"] == 1
    assert summary["worst_total_tensor_ops_rel_error"] == 0.15
    assert summary["worst_max_sm_tensor_ops_rel_error"] == 0.2
