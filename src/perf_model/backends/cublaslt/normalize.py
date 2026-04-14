"""Normalize parsed cuBLAS kernel metadata into repository KernelMeta."""

from __future__ import annotations

from perf_model.common.types import KernelMeta


def build_cublas_kernel_meta(
    parsed: dict[str, int | str | None],
    *,
    dtype: str,
) -> KernelMeta:
    tb_m = int(parsed["threadblock_m"] or 128)
    tb_n = int(parsed["threadblock_n"] or 128)
    tb_k = 32
    return KernelMeta(
        name=str(parsed["kernel_family"]),
        backend="cublas",
        pipeline="tensor_core",
        threadblock_shape=(tb_m, tb_n, tb_k),
        warp_shape=(64, 64, tb_k),
        instruction_shape=(16, 8, 16),
        swizzle="Identity",
        split_k_default=1,
        dtype=dtype,
        extra={
            "stages": parsed.get("stages"),
            "layout_tag": parsed.get("layout_tag"),
            "instruction_family": parsed.get("instruction_family"),
        },
    )
