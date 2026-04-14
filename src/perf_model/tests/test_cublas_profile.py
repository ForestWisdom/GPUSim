import csv

from perf_model.kernel_desc.cublas_empirical import summarize_gemm_call
from perf_model.profiling.cublas_profile import (
    extract_main_kernel_from_nsys_cuda_gpu_trace,
    is_reduction_kernel_name,
    normalize_cublas_profile_row,
    normalize_bench_result,
)


def test_normalize_bench_result_builds_profile_row() -> None:
    row = normalize_bench_result(
        problem={"M": 128, "N": 256, "K": 512},
        bench_result={"latency_us": 17.5, "device": 4, "gpu_name": "RTX 4090"},
        kernel_record={
            "kernel_name": "ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_tn",
            "kernel_index": 0,
            "grid_x": 8,
            "grid_y": 1,
            "grid_z": 1,
            "block_x": 256,
            "block_y": 1,
            "block_z": 1,
        },
        gemm_call_id="call-0",
    )

    assert row["M"] == 128
    assert row["N"] == 256
    assert row["K"] == 512
    assert row["latency_us"] == 17.5
    assert str(row["kernel_name"]).startswith("ampere_fp16")
    assert row["gemm_call_id"] == "call-0"


def test_is_reduction_kernel_name_detects_splitk_reduction() -> None:
    assert is_reduction_kernel_name("void splitKreduceKernel<float>(...)")
    assert not is_reduction_kernel_name("ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_tn")


def test_summarized_rows_can_feed_empirical_summary() -> None:
    rows = [
        normalize_bench_result(
            problem={"M": 128, "N": 128, "K": 128},
            bench_result={"latency_us": 10.0, "device": 4, "gpu_name": "RTX 4090"},
            kernel_record={
                "kernel_name": "main_kernel",
                "kernel_index": 0,
                "grid_x": 4,
                "grid_y": 1,
                "grid_z": 1,
                "block_x": 256,
                "block_y": 1,
                "block_z": 1,
            },
            gemm_call_id="call-0",
        )
    ]

    summary = summarize_gemm_call(rows)

    assert summary["main_kernel_task_count"] == 4


def test_normalize_cublas_profile_row_keeps_kernel_name() -> None:
    row = normalize_cublas_profile_row(
        problem={"M": 128, "N": 256, "K": 512},
        bench_payload={
            "latency_us": 12.0,
            "device": 4,
            "gpu_name": "RTX 4090",
            "kernel_name": "ampere_h16816gemm_128x128_ldg8",
        },
    )

    assert row["kernel_name"] == "ampere_h16816gemm_128x128_ldg8"
    assert row["M"] == 128
    assert row["latency_us"] == 12.0


def test_extract_main_kernel_from_nsys_cuda_gpu_trace_selects_longest_kernel(tmp_path) -> None:
    trace = tmp_path / "trace.csv"
    fieldnames = [
        "Start (ns)",
        "Duration (ns)",
        "CorrId",
        "GrdX",
        "GrdY",
        "GrdZ",
        "BlkX",
        "BlkY",
        "BlkZ",
        "Reg/Trd",
        "StcSMem (MB)",
        "DymSMem (MB)",
        "Bytes (MB)",
        "Throughput (MB/s)",
        "SrcMemKd",
        "DstMemKd",
        "Device",
        "Ctx",
        "GreenCtx",
        "Strm",
        "Name",
    ]
    with trace.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"Start (ns)": 1, "Duration (ns)": 100, "Name": "[CUDA memset]"})
        writer.writerow(
            {
                "Start (ns)": 2,
                "Duration (ns)": 200,
                "CorrId": 1,
                "GrdX": 2,
                "GrdY": 1,
                "GrdZ": 1,
                "BlkX": 128,
                "BlkY": 1,
                "BlkZ": 1,
                "Device": "NVIDIA GeForce RTX 4090 (4)",
                "Ctx": 1,
                "Strm": 7,
                "Name": "short_kernel",
            }
        )
        writer.writerow(
            {
                "Start (ns)": 3,
                "Duration (ns)": 500,
                "CorrId": 2,
                "GrdX": 4,
                "GrdY": 1,
                "GrdZ": 1,
                "BlkX": 256,
                "BlkY": 1,
                "BlkZ": 1,
                "Device": "NVIDIA GeForce RTX 4090 (4)",
                "Ctx": 1,
                "Strm": 7,
                "Name": "long_kernel",
            }
        )

    kernel = extract_main_kernel_from_nsys_cuda_gpu_trace(trace)

    assert kernel["kernel_name"] == "long_kernel"
    assert kernel["grid_x"] == 4
    assert kernel["block_x"] == 256
