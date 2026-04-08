"""Helpers to assemble flat model features."""

from __future__ import annotations

from perf_model.common.types import GemmProblem, GpuSpec, KernelMeta, SmFeatures


def build_feature_vector(
    problem: GemmProblem,
    gpu: GpuSpec,
    kernel_meta: KernelMeta,
    aggregated_gpu_features: list[float],
) -> list[float]:
    return [
        float(problem.M),
        float(problem.N),
        float(problem.K),
        float(problem.split_k_slices),
        float(gpu.num_sms),
        float(gpu.clock_mhz),
        float(kernel_meta.threadblock_shape[0]),
        float(kernel_meta.threadblock_shape[1]),
        float(kernel_meta.threadblock_shape[2]),
        *aggregated_gpu_features,
    ]


def summarize_sm_features(sm_features: list[SmFeatures]) -> dict[str, float]:
    if not sm_features:
        return {"active_sms": 0.0, "max_busy_cycles": 0.0}
    return {
        "active_sms": float(sum(1 for item in sm_features if item.task_count > 0)),
        "max_busy_cycles": float(max(item.estimated_busy_cycles for item in sm_features)),
    }
