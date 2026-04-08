"""Standalone memory traffic helpers."""

from __future__ import annotations

from perf_model.common.constants import DTYPE_BYTES
from perf_model.common.types import GemmTask, KernelMeta


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
