from __future__ import annotations

from pathlib import Path

import yaml

from scripts.build_dataset import load_gpu_spec
from perf_model.common.types import GemmTask, GpuSpec, KernelMeta
from perf_model.features.gemm_tensor_core import TensorCoreFeatureBuilder


def test_load_gpu_spec_reads_bytes_per_cycle_bandwidth_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "gpu.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "toy",
                "num_sms": 1,
                "tensor_throughput_per_sm": 64.0,
                "simt_throughput_per_sm": 32.0,
                "dram_bw_bytes_per_cycle": 400.0,
                "l2_bw_bytes_per_cycle": 800.0,
                "smem_bw_bytes_per_cycle_per_sm": 160.0,
                "clock_mhz": 1000.0,
            }
        )
    )

    gpu = load_gpu_spec(str(config_path))

    assert gpu.dram_bw_bytes_per_cycle == 400.0
    assert gpu.l2_bw_bytes_per_cycle == 800.0
    assert gpu.smem_bw_bytes_per_cycle_per_sm == 160.0


def test_tensor_core_feature_builder_uses_bytes_per_cycle_bandwidth() -> None:
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
        dram_bw_bytes_per_cycle=16.0,
        l2_bw_bytes_per_cycle=32.0,
        smem_bw_bytes_per_cycle_per_sm=64.0,
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

    assert features.bytes_global == 448.0
    assert features.global_cycles == 28.0
    assert features.l2_cycles == 14.0
    assert features.smem_cycles == 3.0
