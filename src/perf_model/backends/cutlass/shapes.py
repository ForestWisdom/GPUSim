"""Threadblock / warp / instruction shape helpers."""

from __future__ import annotations

from perf_model.common.types import KernelMeta


def shape_summary(kernel_meta: KernelMeta) -> dict[str, tuple[int, int, int]]:
    return {
        "threadblock_shape": kernel_meta.threadblock_shape,
        "warp_shape": kernel_meta.warp_shape,
        "instruction_shape": kernel_meta.instruction_shape,
    }
