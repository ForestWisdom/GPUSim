from perf_model.common.types import GemmProblem, GpuSpec, KernelMeta, SmFeatures
from perf_model.features.feature_vector import (
    FEATURE_VECTOR_FIELDS,
    build_feature_vector,
    get_feature_column_name,
    summarize_sm_features,
)


def test_summarize_sm_features_keeps_memory_hierarchy_separate() -> None:
    sm_features = [
        SmFeatures(
            sm_id=0,
            task_count=2,
            total_tensor_ops=100.0,
            total_tensor_cycles=10.0,
            total_bytes_global_raw=260.0,
            total_bytes_global=200.0,
            total_global_cycles=20.0,
            total_l2_cycles=12.0,
            total_smem_cycles=8.0,
            estimated_busy_cycles=20.0,
            reuse_a_factor=1.5,
            reuse_b_factor=1.0,
        ),
        SmFeatures(
            sm_id=1,
            task_count=1,
            total_tensor_ops=40.0,
            total_tensor_cycles=4.0,
            total_bytes_global_raw=90.0,
            total_bytes_global=80.0,
            total_global_cycles=8.0,
            total_l2_cycles=6.0,
            total_smem_cycles=2.0,
            estimated_busy_cycles=8.0,
            reuse_a_factor=1.0,
            reuse_b_factor=1.0,
        ),
    ]

    summary = summarize_sm_features(sm_features)

    assert summary["gpu_total_global_cycles"] == 28.0
    assert summary["gpu_total_l2_cycles"] == 18.0
    assert summary["gpu_total_smem_cycles"] == 10.0
    assert summary["gpu_total_bytes_global_raw"] == 350.0
    assert summary["max_sm_global_cycles"] == 20.0
    assert summary["max_sm_l2_cycles"] == 12.0
    assert summary["max_sm_smem_cycles"] == 8.0
    assert summary["avg_reuse_a_factor"] == 1.25
    assert summary["avg_reuse_b_factor"] == 1.0


def test_build_feature_vector_assigns_stable_swizzle_ids_for_identity_n() -> None:
    gpu = GpuSpec(
        name="toy",
        num_sms=2,
        tensor_throughput_per_sm=64.0,
        simt_throughput_per_sm=32.0,
        dram_bw_bytes_per_cycle=100.0,
        l2_bw_bytes_per_cycle=200.0,
        smem_bw_bytes_per_cycle_per_sm=300.0,
        clock_mhz=1000.0,
    )
    kernel = KernelMeta(
        name="cutlass_tensorop",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(128, 128, 32),
        warp_shape=(64, 64, 32),
        instruction_shape=(16, 8, 16),
        swizzle="Identity4",
    )

    vector = build_feature_vector(
        problem=GemmProblem(M=128, N=128, K=64, split_k_slices=1),
        gpu=gpu,
        kernel_meta=kernel,
        aggregated_gpu_features=[1.0, 2.0, 3.0],
    )

    assert vector[21] == 0.5


def test_feature_vector_exposes_named_column_for_max_sm_busy_cycles() -> None:
    assert FEATURE_VECTOR_FIELDS.index("max_sm_busy_cycles") == 33
    assert get_feature_column_name("max_sm_busy_cycles") == "f_33"
