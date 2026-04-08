"""SIMT GEMM analytical features."""

from __future__ import annotations

from perf_model.common.constants import DTYPE_BYTES
from perf_model.common.types import GemmTask, GpuSpec, KernelMeta, SmFeatures, TaskFeatures
from perf_model.features.base import FeatureBuilder


class SimtFeatureBuilder(FeatureBuilder):
    def build_task_features(
        self, task: GemmTask, sm_id: int, gpu: GpuSpec, kernel_meta: KernelMeta
    ) -> TaskFeatures:
        dtype_bytes = DTYPE_BYTES.get(kernel_meta.dtype, 4)
        math_ops = float(2 * task.m_eff * task.n_eff * task.k_eff)
        math_cycles = math_ops / max(gpu.simt_throughput_per_sm, 1.0)
        bytes_a = float(task.m_eff * task.k_eff * dtype_bytes)
        bytes_b = float(task.k_eff * task.n_eff * dtype_bytes)
        bytes_c = float(task.m_eff * task.n_eff * dtype_bytes)
        memory_cycles = (bytes_a + bytes_b + bytes_c) / max(gpu.smem_bw_gbps_per_sm, 1.0)
        return TaskFeatures(
            task_idx=task.task_idx,
            sm_id=sm_id,
            math_ops=math_ops,
            math_cycles=math_cycles,
            bytes_a=bytes_a,
            bytes_b=bytes_b,
            bytes_c=bytes_c,
            memory_cycles=memory_cycles,
        )

    def aggregate_sm_features(self, sm_id: int, task_features: list[TaskFeatures]) -> SmFeatures:
        total_math_cycles = sum(item.math_cycles for item in task_features)
        total_memory_cycles = sum(item.memory_cycles for item in task_features)
        return SmFeatures(
            sm_id=sm_id,
            task_count=len(task_features),
            total_math_ops=sum(item.math_ops for item in task_features),
            total_math_cycles=total_math_cycles,
            total_memory_cycles=total_memory_cycles,
            estimated_busy_cycles=max(total_math_cycles, total_memory_cycles),
        )

    def aggregate_gpu_features(self, sm_features: list[SmFeatures]) -> list[float]:
        if not sm_features:
            return [0.0] * 8
        return [
            float(sum(item.total_math_ops for item in sm_features)),
            float(sum(item.total_math_cycles for item in sm_features)),
            float(sum(item.total_memory_cycles for item in sm_features)),
            float(max(item.estimated_busy_cycles for item in sm_features)),
            float(sum(item.task_count for item in sm_features)),
            float(len(sm_features)),
            0.0,
            0.0,
        ]
