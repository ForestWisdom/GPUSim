"""Standalone memory traffic helpers."""

from __future__ import annotations

from perf_model.common.constants import DTYPE_BYTES
from perf_model.common.types import GemmTask, KernelMeta, TaskFeatures


def estimate_task_memory_bytes(task: GemmTask, kernel_meta: KernelMeta) -> dict[str, float]:
    dtype_bytes = DTYPE_BYTES.get(kernel_meta.dtype, 2)
    bytes_a = float(task.m_eff * task.k_eff * dtype_bytes)
    bytes_b = float(task.k_eff * task.n_eff * dtype_bytes)
    bytes_c = float(task.m_eff * task.n_eff * dtype_bytes)
    return {
        "bytes_a": bytes_a,
        "bytes_b": bytes_b,
        "bytes_c": bytes_c,
        "bytes_total": bytes_a + bytes_b + bytes_c,
    }


def estimate_same_sm_memory_reuse(task_features: list[TaskFeatures]) -> dict[str, float]:
    if not task_features:
        return {
            "total_bytes_global_raw": 0.0,
            "total_bytes_global_effective": 0.0,
            "unique_bytes_a": 0.0,
            "unique_bytes_b": 0.0,
            "total_bytes_c": 0.0,
            "reuse_a_factor": 1.0,
            "reuse_b_factor": 1.0,
        }

    total_bytes_a = sum(item.bytes_a for item in task_features)
    total_bytes_b = sum(item.bytes_b for item in task_features)
    total_bytes_c = sum(item.bytes_c for item in task_features)
    total_bytes_global_raw = sum(item.bytes_global for item in task_features)

    unique_a_panels: dict[tuple[int, int], float] = {}
    unique_b_panels: dict[tuple[int, int], float] = {}
    for item in task_features:
        unique_a_panels.setdefault((item.tile_idx_m, item.tile_idx_k), item.bytes_a)
        unique_b_panels.setdefault((item.tile_idx_n, item.tile_idx_k), item.bytes_b)

    unique_bytes_a = sum(unique_a_panels.values())
    unique_bytes_b = sum(unique_b_panels.values())
    total_bytes_global_effective = unique_bytes_a + unique_bytes_b + total_bytes_c

    return {
        "total_bytes_global_raw": total_bytes_global_raw,
        "total_bytes_global_effective": total_bytes_global_effective,
        "unique_bytes_a": unique_bytes_a,
        "unique_bytes_b": unique_bytes_b,
        "total_bytes_c": total_bytes_c,
        "reuse_a_factor": total_bytes_a / max(unique_bytes_a, 1e-6),
        "reuse_b_factor": total_bytes_b / max(unique_bytes_b, 1e-6),
    }
