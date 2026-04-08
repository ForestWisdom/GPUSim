"""Helpers to assemble flat model features."""
from __future__ import annotations

from perf_model.common.types import GemmProblem, GpuSpec, KernelMeta, SmFeatures


_SWIZZLE_TO_ID = {
    "Identity": 0.0,
    "SplitKIdentity": 1.0,
    "Horizontal": 2.0,
}


def build_feature_vector(
    problem: GemmProblem,
    gpu: GpuSpec,
    kernel_meta: KernelMeta,
    aggregated_gpu_features: list[float],
) -> list[float]:
    tb_m, tb_n, tb_k = kernel_meta.threadblock_shape
    wp_m, wp_n, wp_k = kernel_meta.warp_shape
    inst_m, inst_n, inst_k = kernel_meta.instruction_shape

    return [
        # Problem
        float(problem.M),
        float(problem.N),
        float(problem.K),
        float(problem.split_k_slices),
        # GPU
        float(gpu.num_sms),
        float(gpu.clock_mhz),
        float(gpu.tensor_throughput_per_sm),
        float(gpu.simt_throughput_per_sm),
        float(gpu.dram_bw_gbps),
        float(gpu.l2_bw_gbps),
        float(gpu.smem_bw_gbps_per_sm),
        # Kernel meta
        float(tb_m),
        float(tb_n),
        float(tb_k),
        float(wp_m),
        float(wp_n),
        float(wp_k),
        float(inst_m),
        float(inst_n),
        float(inst_k),
        float(kernel_meta.split_k_default),
        float(_SWIZZLE_TO_ID.get(kernel_meta.swizzle, -1.0)),
        # Aggregated analytical features
        *aggregated_gpu_features,
    ]


def summarize_sm_features(sm_features: list[SmFeatures]) -> dict[str, float]:
    if not sm_features:
        return {"active_sms": 0.0, "max_busy_cycles": 0.0}
    return {
        "active_sms": float(sum(1 for item in sm_features if item.task_count > 0)),
        "max_busy_cycles": float(max(item.estimated_busy_cycles for item in sm_features)),
    }