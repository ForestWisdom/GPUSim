from perf_model.backends.cublaslt.normalize import build_cublas_kernel_meta
from perf_model.common.types import GemmProblem, GpuSpec
from perf_model.pipelines.feature_pipeline import build_default_feature_pipeline


def test_build_cublas_kernel_meta_maps_parsed_fields() -> None:
    parsed = {
        "kernel_family": "ampere_h16816gemm",
        "threadblock_m": 128,
        "threadblock_n": 64,
        "stages": 32,
        "layout_tag": "nt",
        "instruction_family": "tensor_core",
    }

    meta = build_cublas_kernel_meta(parsed, dtype="f16")

    assert meta.backend == "cublas"
    assert meta.pipeline == "tensor_core"
    assert meta.threadblock_shape[0] == 128
    assert meta.threadblock_shape[1] == 64
    assert meta.extra["stages"] == 32


def test_normalized_cublas_kernel_meta_drives_existing_pipeline() -> None:
    parsed = {
        "kernel_family": "ampere_h16816gemm",
        "threadblock_m": 128,
        "threadblock_n": 128,
        "stages": 32,
        "layout_tag": "nn",
        "instruction_family": "tensor_core",
    }
    meta = build_cublas_kernel_meta(parsed, dtype="f16")
    gpu = GpuSpec(
        name="RTX4090",
        num_sms=128,
        tensor_throughput_per_sm=165.0,
        simt_throughput_per_sm=128.0,
        dram_bw_bytes_per_cycle=400.0,
        l2_bw_bytes_per_cycle=1000.0,
        smem_bw_bytes_per_cycle_per_sm=256.0,
        clock_mhz=2520.0,
    )
    pipeline = build_default_feature_pipeline(meta)
    problem = GemmProblem(M=256, N=256, K=128)

    feature_vector, _context = pipeline.run(problem, gpu, meta)

    assert len(feature_vector) > 0
