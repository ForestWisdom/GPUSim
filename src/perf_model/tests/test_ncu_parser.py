from __future__ import annotations

from pathlib import Path

from perf_model.profiling.ncu_parser import extract_metric_summary, parse_ncu_report


def test_parse_ncu_report_extracts_metric_values_and_instances(tmp_path: Path) -> None:
    csv_path = tmp_path / "ncu.csv"
    csv_path.write_text(
        "\n".join(
            [
                '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value","Metric Instances"',
                '"0","cutlass_kernel","sm__ops_path_tensor_src_fp16_dst_fp16.sum","ops","2048","512;1024;512"',
                '"1","cutlass_kernel","gpu__time_duration.sum","ns","1000",""',
            ]
        ),
        encoding="utf-8",
    )

    report = parse_ncu_report(csv_path)
    summary = extract_metric_summary(report, "sm__ops_path_tensor_src_fp16_dst_fp16.sum")

    assert summary.metric_name == "sm__ops_path_tensor_src_fp16_dst_fp16.sum"
    assert summary.total == 2048.0
    assert summary.max_instance == 1024.0
    assert summary.instances == [512.0, 1024.0, 512.0]
    assert report.kernel_names == ["cutlass_kernel"]


def test_extract_metric_summary_aggregates_multiple_kernel_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "ncu.csv"
    csv_path.write_text(
        "\n".join(
            [
                '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value","Metric Instances"',
                '"0","cutlass_kernel_0","sm__ops_path_tensor_src_fp16_dst_fp16.sum","ops","2048","512;1024;512"',
                '"1","cutlass_kernel_1","sm__ops_path_tensor_src_fp16_dst_fp16.sum","ops","1024","256;768"',
            ]
        ),
        encoding="utf-8",
    )

    report = parse_ncu_report(csv_path)
    summary = extract_metric_summary(report, "sm__ops_path_tensor_src_fp16_dst_fp16.sum")

    assert summary.total == 3072.0
    assert summary.max_instance == 1024.0
    assert summary.instance_count == 5


def test_parse_ncu_report_supports_wide_raw_csv_format(tmp_path: Path) -> None:
    csv_path = tmp_path / "ncu_raw.csv"
    csv_path.write_text(
        "\n".join(
            [
                "==PROF== Connected to process 1",
                '"ID","Kernel Name","gpu__time_duration.sum","smsp__inst_executed_pipe_tensor_op_hmma_v2.sum"',
                '"","","ns","inst"',
                '"0","cutlass_kernel","100","256 (64;64;64;64)"',
                '"1","cutlass_kernel","120","128 (32;32;32;32)"',
            ]
        ),
        encoding="utf-8",
    )

    report = parse_ncu_report(csv_path)
    summary = extract_metric_summary(report, "smsp__inst_executed_pipe_tensor_op_hmma_v2.sum")

    assert report.kernel_names == ["cutlass_kernel"]
    assert summary.total == 384.0
    assert summary.max_instance == 64.0
    assert summary.instance_count == 8
