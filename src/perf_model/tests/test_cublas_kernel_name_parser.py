from perf_model.backends.cublaslt.kernel_name_parser import parse_cublas_kernel_name


def test_parse_cublas_kernel_name_extracts_threadblock_shape() -> None:
    parsed = parse_cublas_kernel_name("ampere_h16816gemm_128x128_ldg8_stages_32x1_nn")

    assert parsed["threadblock_m"] == 128
    assert parsed["threadblock_n"] == 128
    assert str(parsed["kernel_family"]).startswith("ampere_h16816gemm")


def test_parse_cublas_kernel_name_extracts_stage_count_when_present() -> None:
    parsed = parse_cublas_kernel_name("ampere_h16816gemm_128x64_ldg8_stages_32x1_nt")

    assert parsed["stages"] == 32
    assert parsed["layout_tag"] == "nt"
