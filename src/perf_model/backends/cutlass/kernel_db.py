"""Known CUTLASS kernels used by experiments."""

from __future__ import annotations

from perf_model.common.types import KernelMeta

CUTLASS_KERNELS: dict[str, KernelMeta] = {
    "cutlass_tensorop_128x128x32": KernelMeta(
        name="cutlass_tensorop_128x128x32",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(128, 128, 32),
        warp_shape=(64, 64, 32),
        instruction_shape=(16, 8, 16),
        dtype="f16",
    ),
    "cutlass_simt_128x64x8": KernelMeta(
        name="cutlass_simt_128x64x8",
        backend="cutlass",
        pipeline="simt",
        threadblock_shape=(128, 64, 8),
        warp_shape=(64, 32, 8),
        instruction_shape=(1, 1, 1),
        dtype="f32",
    ),
}
