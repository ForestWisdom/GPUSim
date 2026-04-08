from perf_model.common.types import GemmTask, GpuSpec, KernelMeta
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

    assert features.math_ops == 2 * 8 * 16 * 4


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
