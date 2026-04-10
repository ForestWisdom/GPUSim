"""Helpers to assemble flat model features."""
from __future__ import annotations

from perf_model.common.types import GemmProblem, GpuSpec, KernelMeta, SmFeatures


_SWIZZLE_TO_ID = {
    "Identity": 0.0,
    "Identity2": 0.25,
    "Identity4": 0.5,
    "Identity8": 0.75,
    "SplitKIdentity": 1.0,
    "SplitKIdentity2": 1.25,
    "SplitKIdentity4": 1.5,
    "SplitKIdentity8": 1.75,
    "Horizontal": 2.0,
    "SplitKHorizontal": 2.25,
}

FEATURE_VECTOR_FIELDS = [
    # Problem
    "problem_m",
    "problem_n",
    "problem_k",
    "problem_split_k_slices",
    # GPU
    "gpu_num_sms",
    "gpu_clock_mhz",
    "gpu_tensor_throughput_per_sm",
    "gpu_simt_throughput_per_sm",
    "gpu_dram_bw_bytes_per_cycle",
    "gpu_l2_bw_bytes_per_cycle",
    "gpu_smem_bw_bytes_per_cycle_per_sm",
    # Kernel meta
    "threadblock_m",
    "threadblock_n",
    "threadblock_k",
    "warp_m",
    "warp_n",
    "warp_k",
    "instruction_m",
    "instruction_n",
    "instruction_k",
    "kernel_split_k_default",
    "kernel_swizzle_id",
    # Aggregated analytical features
    "gpu_total_tensor_ops",
    "gpu_total_tensor_cycles",
    "gpu_total_bytes_global",
    "gpu_total_global_cycles",
    "gpu_total_l2_cycles",
    "gpu_total_smem_cycles",
    "max_sm_tensor_ops",
    "max_sm_tensor_cycles",
    "max_sm_global_cycles",
    "max_sm_l2_cycles",
    "max_sm_smem_cycles",
    "max_sm_busy_cycles",
    "avg_sm_busy_cycles",
    "max_task_count",
    "avg_task_count",
    "active_sms",
    "gpu_total_bytes_global_raw",
    "avg_reuse_a_factor",
    "avg_reuse_b_factor",
]


def get_feature_column_name(feature_name: str) -> str:
    return f"f_{FEATURE_VECTOR_FIELDS.index(feature_name)}"


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
        float(gpu.dram_bw_bytes_per_cycle),
        float(gpu.l2_bw_bytes_per_cycle),
        float(gpu.smem_bw_bytes_per_cycle_per_sm),
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
        return {
            "active_sms": 0.0,
            "gpu_total_tensor_ops": 0.0,
            "gpu_total_tensor_cycles": 0.0,
            "gpu_total_bytes_global_raw": 0.0,
            "gpu_total_bytes_global": 0.0,
            "gpu_total_global_cycles": 0.0,
            "gpu_total_l2_cycles": 0.0,
            "gpu_total_smem_cycles": 0.0,
            "max_sm_tensor_ops": 0.0,
            "max_sm_tensor_cycles": 0.0,
            "max_sm_global_cycles": 0.0,
            "max_sm_l2_cycles": 0.0,
            "max_sm_smem_cycles": 0.0,
            "max_busy_cycles": 0.0,
            "avg_sm_busy_cycles": 0.0,
            "max_task_count": 0.0,
            "avg_task_count": 0.0,
            "avg_reuse_a_factor": 1.0,
            "avg_reuse_b_factor": 1.0,
        }
    return {
        "active_sms": float(sum(1 for item in sm_features if item.task_count > 0)),
        "gpu_total_tensor_ops": float(sum(item.total_tensor_ops for item in sm_features)),
        "gpu_total_tensor_cycles": float(sum(item.total_tensor_cycles for item in sm_features)),
        "gpu_total_bytes_global_raw": float(sum(item.total_bytes_global_raw for item in sm_features)),
        "gpu_total_bytes_global": float(sum(item.total_bytes_global for item in sm_features)),
        "gpu_total_global_cycles": float(sum(item.total_global_cycles for item in sm_features)),
        "gpu_total_l2_cycles": float(sum(item.total_l2_cycles for item in sm_features)),
        "gpu_total_smem_cycles": float(sum(item.total_smem_cycles for item in sm_features)),
        "max_sm_tensor_ops": float(max(item.total_tensor_ops for item in sm_features)),
        "max_sm_tensor_cycles": float(max(item.total_tensor_cycles for item in sm_features)),
        "max_sm_global_cycles": float(max(item.total_global_cycles for item in sm_features)),
        "max_sm_l2_cycles": float(max(item.total_l2_cycles for item in sm_features)),
        "max_sm_smem_cycles": float(max(item.total_smem_cycles for item in sm_features)),
        "max_busy_cycles": float(max(item.estimated_busy_cycles for item in sm_features)),
        "avg_sm_busy_cycles": float(
            sum(item.estimated_busy_cycles for item in sm_features) / len(sm_features)
        ),
        "max_task_count": float(max(item.task_count for item in sm_features)),
        "avg_task_count": float(sum(item.task_count for item in sm_features) / len(sm_features)),
        "avg_reuse_a_factor": float(sum(item.reuse_a_factor for item in sm_features) / len(sm_features)),
        "avg_reuse_b_factor": float(sum(item.reuse_b_factor for item in sm_features) / len(sm_features)),
    }
