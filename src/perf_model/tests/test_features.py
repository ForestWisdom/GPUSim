from perf_model.common.types import GemmTask, GpuSpec, KernelMeta
from perf_model.features.gemm_simt import SimtFeatureBuilder
from perf_model.features.gemm_tensor_core import TensorCoreFeatureBuilder
from perf_model.features.memory_model import estimate_task_memory_bytes


def test_tensor_core_math_ops_matches_2mnk() -> None:
    task = GemmTask(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        m0=0,
        m1=8,
        n0=0,
        n1=16,
        k0=0,
        k1=4,
        m_eff=8,
        n_eff=16,
        k_eff=4,
        gemm_k_iterations=1,
        task_idx=0,
        tile_idx_m=0,
        tile_idx_n=0,
        tile_idx_k=0,
    )
    gpu = GpuSpec(
        name="toy",
        num_sms=1,
        tensor_throughput_per_sm=64.0,
        simt_throughput_per_sm=32.0,
        dram_bw_gbps=1.0,
        l2_bw_gbps=1.0,
        smem_bw_gbps_per_sm=128.0,
        clock_mhz=1000.0,
    )
    kernel = KernelMeta(
        name="cutlass_tensorop",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(128, 128, 32),
        warp_shape=(64, 64, 32),
        instruction_shape=(16, 8, 16),
    )

    features = TensorCoreFeatureBuilder().build_task_features(task, 0, gpu, kernel)

    assert features.tensor_ops == 2 * 8 * 16 * 4
    assert features.tensor_cycles == (2 * 8 * 16 * 4) / 64.0


def test_memory_model_estimates_bytes() -> None:
    task = GemmTask(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        m0=0,
        m1=8,
        n0=0,
        n1=16,
        k0=0,
        k1=4,
        m_eff=8,
        n_eff=16,
        k_eff=4,
        gemm_k_iterations=1,
        task_idx=0,
        tile_idx_m=0,
        tile_idx_n=0,
        tile_idx_k=0,
    )
    kernel = KernelMeta(
        name="cutlass_tensorop",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(128, 128, 32),
        warp_shape=(64, 64, 32),
        instruction_shape=(16, 8, 16),
        dtype="f16",
    )

    memory = estimate_task_memory_bytes(task, kernel)

    assert memory["bytes_a"] == 64.0
    assert memory["bytes_b"] == 128.0
    assert memory["bytes_c"] == 256.0


def test_simt_feature_builder_matches_new_task_feature_schema() -> None:
    task = GemmTask(
        tile_m=128,
        tile_n=64,
        tile_k=8,
        m0=0,
        m1=32,
        n0=0,
        n1=16,
        k0=0,
        k1=8,
        m_eff=32,
        n_eff=16,
        k_eff=8,
        gemm_k_iterations=1,
        task_idx=0,
        tile_idx_m=0,
        tile_idx_n=0,
        tile_idx_k=0,
    )
    gpu = GpuSpec(
        name="toy",
        num_sms=1,
        tensor_throughput_per_sm=64.0,
        simt_throughput_per_sm=32.0,
        dram_bw_gbps=64.0,
        l2_bw_gbps=128.0,
        smem_bw_gbps_per_sm=256.0,
        clock_mhz=1000.0,
    )
    kernel = KernelMeta(
        name="cutlass_simt",
        backend="cutlass",
        pipeline="simt",
        threadblock_shape=(128, 64, 8),
        warp_shape=(64, 32, 8),
        instruction_shape=(1, 1, 1),
        dtype="f32",
    )

    builder = SimtFeatureBuilder()
    task_features = builder.build_task_features(task, 0, gpu, kernel)
    sm_features = builder.aggregate_sm_features(0, [task_features])
    gpu_features = builder.aggregate_gpu_features([sm_features])

    assert task_features.tensor_ops == 2 * 32 * 16 * 8
    assert task_features.bytes_global > 0.0
    assert sm_features.total_tensor_ops == task_features.tensor_ops
    assert sm_features.estimated_busy_cycles >= task_features.tensor_cycles
    assert len(gpu_features) > 0
