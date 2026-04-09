from __future__ import annotations

from pathlib import Path

from perf_model.validation.ncu_compare import build_ncu_tensor_summary


def test_build_ncu_tensor_summary_keeps_single_task_total_as_max_sm(tmp_path: Path) -> None:
    csv_path = tmp_path / "ncu_single_task.csv"
    csv_path.write_text(
        "\n".join(
            [
                "==PROF== Connected to process 1",
                '"ID","Kernel Name","smsp__inst_executed_pipe_tensor_op_hmma_v2.sum"',
                '"","","inst"',
                '"0","cutlass_kernel","1024"',
            ]
        ),
        encoding="utf-8",
    )

    summary = build_ncu_tensor_summary(
        csv_path,
        metric_name="smsp__inst_executed_pipe_tensor_op_hmma_v2.sum",
        metric_scale=4096.0,
        model_task_count=1,
    )

    assert summary.total_tensor_ops == 4194304.0
    assert summary.max_sm_tensor_ops == 4194304.0
    assert summary.max_sm_supported is True
    assert summary.max_sm_source == "single_task_total"


def test_build_ncu_tensor_summary_marks_max_sm_unknown_for_multi_task_aggregate_metric(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "ncu_multi_task.csv"
    csv_path.write_text(
        "\n".join(
            [
                "==PROF== Connected to process 1",
                '"ID","Kernel Name","smsp__inst_executed_pipe_tensor_op_hmma_v2.sum"',
                '"","","inst"',
                '"0","cutlass_kernel","4096"',
            ]
        ),
        encoding="utf-8",
    )

    summary = build_ncu_tensor_summary(
        csv_path,
        metric_name="smsp__inst_executed_pipe_tensor_op_hmma_v2.sum",
        metric_scale=4096.0,
        model_task_count=4,
    )

    assert summary.total_tensor_ops == 16777216.0
    assert summary.max_sm_tensor_ops is None
    assert summary.max_sm_supported is False
    assert summary.max_sm_source == "aggregate_only"


def test_build_ncu_tensor_summary_uses_dedicated_max_metric_when_provided(tmp_path: Path) -> None:
    csv_path = tmp_path / "ncu_multi_task_with_max.csv"
    csv_path.write_text(
        "\n".join(
            [
                "==PROF== Connected to process 1",
                '"ID","Kernel Name","sm__inst_executed_pipe_tensor_op_hmma_v2.sum","sm__inst_executed_pipe_tensor_op_hmma_v2.max"',
                '"","","inst","inst"',
                '"0","cutlass_kernel","4096","1024"',
            ]
        ),
        encoding="utf-8",
    )

    summary = build_ncu_tensor_summary(
        csv_path,
        metric_name="sm__inst_executed_pipe_tensor_op_hmma_v2.sum",
        metric_scale=4096.0,
        max_metric_name="sm__inst_executed_pipe_tensor_op_hmma_v2.max",
        max_metric_scale=4096.0,
        model_task_count=4,
    )

    assert summary.total_tensor_ops == 16777216.0
    assert summary.max_sm_tensor_ops == 4194304.0
    assert summary.max_sm_supported is True
    assert summary.max_sm_source == "metric:sm__inst_executed_pipe_tensor_op_hmma_v2.max"
