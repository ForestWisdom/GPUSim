"""Tensor Core GEMM analytical features."""
from __future__ import annotations

from perf_model.common.constants import DTYPE_BYTES
from perf_model.common.types import GemmTask, GpuSpec, KernelMeta, SmFeatures, TaskFeatures
from perf_model.features.base import FeatureBuilder
from perf_model.features.memory_model import estimate_same_sm_memory_reuse


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


class TensorCoreFeatureBuilder(FeatureBuilder):
    """A simplified SYNPERF-style analytical feature builder for Tensor Core GEMM."""

    def build_task_features(
        self,
        task: GemmTask,
        sm_id: int,
        gpu: GpuSpec,
        kernel_meta: KernelMeta,
    ) -> TaskFeatures:
        dtype_bytes = float(DTYPE_BYTES.get(kernel_meta.dtype, 2))

        # GEMM math work on Tensor pipeline
        tensor_ops = float(2 * task.m_eff * task.n_eff * task.k_eff)
        tensor_cycles = _safe_div(tensor_ops, gpu.tensor_throughput_per_sm)

        # Coarse-grained memory demand
        bytes_a = float(task.m_eff * task.k_eff * dtype_bytes)
        bytes_b = float(task.k_eff * task.n_eff * dtype_bytes)
        bytes_c = float(task.m_eff * task.n_eff * dtype_bytes)

        # A very simple forward GEMM approximation:
        # read A/B, read+write C/D once
        bytes_global = bytes_a + bytes_b + 2.0 * bytes_c

        global_cycles = _safe_div(bytes_global, gpu.dram_bw_bytes_per_cycle)
        l2_cycles = _safe_div(bytes_global, gpu.l2_bw_bytes_per_cycle)

        # For shared memory, first-order approximation:
        # A/B tiles are staged through shared memory
        smem_cycles = _safe_div(bytes_a + bytes_b, gpu.smem_bw_bytes_per_cycle_per_sm)

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
        total_tensor_ops = sum(item.tensor_ops for item in task_features)
        total_tensor_cycles = sum(item.tensor_cycles for item in task_features)

        memory_reuse = estimate_same_sm_memory_reuse(task_features)
        total_bytes_global_raw = memory_reuse["total_bytes_global_raw"]
        total_bytes_global = memory_reuse["total_bytes_global_effective"]
        raw_global_cycles = sum(item.global_cycles for item in task_features)
        raw_l2_cycles = sum(item.l2_cycles for item in task_features)
        total_global_cycles = raw_global_cycles * _safe_div(
            total_bytes_global, total_bytes_global_raw
        )
        total_l2_cycles = raw_l2_cycles
        total_smem_cycles = sum(item.smem_cycles for item in task_features)

        estimated_busy_cycles = max(
            total_tensor_cycles,
            total_global_cycles,
            total_l2_cycles,
            total_smem_cycles,
        )

        return SmFeatures(
            sm_id=sm_id,
            task_count=len(task_features),
            total_tensor_ops=total_tensor_ops,
            total_tensor_cycles=total_tensor_cycles,
            total_bytes_global_raw=total_bytes_global_raw,
            total_bytes_global=total_bytes_global,
            total_global_cycles=total_global_cycles,
            total_l2_cycles=total_l2_cycles,
            total_smem_cycles=total_smem_cycles,
            estimated_busy_cycles=estimated_busy_cycles,
            reuse_a_factor=memory_reuse["reuse_a_factor"],
            reuse_b_factor=memory_reuse["reuse_b_factor"],
        )

    def aggregate_gpu_features(self, sm_features: list[SmFeatures]) -> list[float]:
        if not sm_features:
            return [0.0] * 19

        gpu_total_tensor_ops = float(sum(item.total_tensor_ops for item in sm_features))
        gpu_total_tensor_cycles_sum = float(sum(item.total_tensor_cycles for item in sm_features))

        gpu_total_bytes_global_raw = float(sum(item.total_bytes_global_raw for item in sm_features))
        gpu_total_bytes_global = float(sum(item.total_bytes_global for item in sm_features))
        gpu_total_global_cycles_sum = float(sum(item.total_global_cycles for item in sm_features))
        gpu_total_l2_cycles_sum = float(sum(item.total_l2_cycles for item in sm_features))
        gpu_total_smem_cycles_sum = float(sum(item.total_smem_cycles for item in sm_features))

        max_sm_tensor_ops = float(max(item.total_tensor_ops for item in sm_features))
        max_sm_tensor_cycles = float(max(item.total_tensor_cycles for item in sm_features))
        max_sm_global_cycles = float(max(item.total_global_cycles for item in sm_features))
        max_sm_l2_cycles = float(max(item.total_l2_cycles for item in sm_features))
        max_sm_smem_cycles = float(max(item.total_smem_cycles for item in sm_features))
        max_sm_busy_cycles = float(max(item.estimated_busy_cycles for item in sm_features))

        avg_sm_busy_cycles = float(
            sum(item.estimated_busy_cycles for item in sm_features) / len(sm_features)
        )
        max_task_count = float(max(item.task_count for item in sm_features))
        avg_task_count = float(sum(item.task_count for item in sm_features) / len(sm_features))
        active_sms = float(sum(1 for item in sm_features if item.task_count > 0))
        avg_reuse_a_factor = float(sum(item.reuse_a_factor for item in sm_features) / len(sm_features))
        avg_reuse_b_factor = float(sum(item.reuse_b_factor for item in sm_features) / len(sm_features))

        return [
            # GPU total
            gpu_total_tensor_ops,
            gpu_total_tensor_cycles_sum,
            gpu_total_bytes_global,
            gpu_total_global_cycles_sum,
            gpu_total_l2_cycles_sum,
            gpu_total_smem_cycles_sum,
            # Max-SM
            max_sm_tensor_ops,
            max_sm_tensor_cycles,
            max_sm_global_cycles,
            max_sm_l2_cycles,
            max_sm_smem_cycles,
            max_sm_busy_cycles,
            # Balance / occupancy-ish
            avg_sm_busy_cycles,
            max_task_count,
            avg_task_count,
            active_sms,
            gpu_total_bytes_global_raw,
            avg_reuse_a_factor,
            avg_reuse_b_factor,
        ]
