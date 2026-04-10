"""SIMT GEMM analytical features."""

from __future__ import annotations

from perf_model.common.constants import DTYPE_BYTES
from perf_model.common.types import GemmTask, GpuSpec, KernelMeta, SmFeatures, TaskFeatures
from perf_model.features.base import FeatureBuilder
from perf_model.features.memory_model import estimate_same_sm_memory_reuse


class SimtFeatureBuilder(FeatureBuilder):
    def build_task_features(
        self, task: GemmTask, sm_id: int, gpu: GpuSpec, kernel_meta: KernelMeta
    ) -> TaskFeatures:
        dtype_bytes = float(DTYPE_BYTES.get(kernel_meta.dtype, 4))
        tensor_ops = float(2 * task.m_eff * task.n_eff * task.k_eff)
        tensor_cycles = tensor_ops / max(gpu.simt_throughput_per_sm, 1.0)
        bytes_a = float(task.m_eff * task.k_eff * dtype_bytes)
        bytes_b = float(task.k_eff * task.n_eff * dtype_bytes)
        bytes_c = float(task.m_eff * task.n_eff * dtype_bytes)
        bytes_global = bytes_a + bytes_b + 2.0 * bytes_c
        global_cycles = bytes_global / max(gpu.dram_bw_bytes_per_cycle, 1.0)
        l2_cycles = bytes_global / max(gpu.l2_bw_bytes_per_cycle, 1.0)
        smem_cycles = (bytes_a + bytes_b) / max(gpu.smem_bw_bytes_per_cycle_per_sm, 1.0)
        return TaskFeatures(
            task_idx=task.task_idx,
            sm_id=sm_id,
            tile_idx_m=task.tile_idx_m,
            tile_idx_n=task.tile_idx_n,
            tile_idx_k=task.tile_idx_k,
            tensor_ops=tensor_ops,
            tensor_cycles=tensor_cycles,
            bytes_a=bytes_a,
            bytes_b=bytes_b,
            bytes_c=bytes_c,
            bytes_global=bytes_global,
            global_cycles=global_cycles,
            l2_cycles=l2_cycles,
            smem_cycles=smem_cycles,
        )

    def aggregate_sm_features(self, sm_id: int, task_features: list[TaskFeatures]) -> SmFeatures:
        memory_reuse = estimate_same_sm_memory_reuse(task_features)
        total_bytes_global_raw = memory_reuse["total_bytes_global_raw"]
        total_bytes_global = memory_reuse["total_bytes_global_effective"]
        total_tensor_cycles = sum(item.tensor_cycles for item in task_features)
        raw_global_cycles = sum(item.global_cycles for item in task_features)
        total_global_cycles = raw_global_cycles * (
            total_bytes_global / max(total_bytes_global_raw, 1e-6)
        )
        total_l2_cycles = sum(item.l2_cycles for item in task_features)
        total_smem_cycles = sum(item.smem_cycles for item in task_features)
        return SmFeatures(
            sm_id=sm_id,
            task_count=len(task_features),
            total_tensor_ops=sum(item.tensor_ops for item in task_features),
            total_tensor_cycles=total_tensor_cycles,
            total_bytes_global_raw=total_bytes_global_raw,
            total_bytes_global=total_bytes_global,
            total_global_cycles=total_global_cycles,
            total_l2_cycles=total_l2_cycles,
            total_smem_cycles=total_smem_cycles,
            estimated_busy_cycles=max(
                total_tensor_cycles,
                total_global_cycles,
                total_l2_cycles,
                total_smem_cycles,
            ),
            reuse_a_factor=memory_reuse["reuse_a_factor"],
            reuse_b_factor=memory_reuse["reuse_b_factor"],
        )

    def aggregate_gpu_features(self, sm_features: list[SmFeatures]) -> list[float]:
        if not sm_features:
            return [0.0] * 19
        return [
            float(sum(item.total_tensor_ops for item in sm_features)),
            float(sum(item.total_tensor_cycles for item in sm_features)),
            float(sum(item.total_bytes_global for item in sm_features)),
            float(sum(item.total_global_cycles for item in sm_features)),
            float(sum(item.total_l2_cycles for item in sm_features)),
            float(sum(item.total_smem_cycles for item in sm_features)),
            float(max(item.total_tensor_ops for item in sm_features)),
            float(max(item.total_tensor_cycles for item in sm_features)),
            float(max(item.total_global_cycles for item in sm_features)),
            float(max(item.total_l2_cycles for item in sm_features)),
            float(max(item.total_smem_cycles for item in sm_features)),
            float(max(item.estimated_busy_cycles for item in sm_features)),
            float(sum(item.estimated_busy_cycles for item in sm_features) / len(sm_features)),
            float(max(item.task_count for item in sm_features)),
            float(sum(item.task_count for item in sm_features) / len(sm_features)),
            float(sum(1 for item in sm_features if item.task_count > 0)),
            float(sum(item.total_bytes_global_raw for item in sm_features)),
            float(sum(item.reuse_a_factor for item in sm_features) / len(sm_features)),
            float(sum(item.reuse_b_factor for item in sm_features) / len(sm_features)),
        ]
