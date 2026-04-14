from perf_model.kernel_desc.cublas_empirical import kernel_family_name, summarize_gemm_call


def test_kernel_family_name_collapses_numeric_detail() -> None:
    name = "ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_tn"

    family = kernel_family_name(name)

    assert family.startswith("ampere_fp16_s16816gemm_fp16")
    assert "128x128" not in family


def test_summarize_gemm_call_extracts_main_kernel_summary() -> None:
    rows = [
        {
            "kernel_name": "ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_tn",
            "kernel_index": 0,
            "grid_x": 12,
            "grid_y": 1,
            "grid_z": 1,
            "block_x": 256,
            "block_y": 1,
            "block_z": 1,
            "is_reduction_kernel": False,
        },
        {
            "kernel_name": "void splitKreduceKernel<float>(...)",
            "kernel_index": 1,
            "grid_x": 3,
            "grid_y": 1,
            "grid_z": 1,
            "block_x": 256,
            "block_y": 1,
            "block_z": 1,
            "is_reduction_kernel": True,
        },
    ]

    summary = summarize_gemm_call(rows)

    assert summary["has_reduction_kernel"] is True
    assert summary["main_kernel_task_count"] == 12
    assert summary["total_kernel_count"] == 2
