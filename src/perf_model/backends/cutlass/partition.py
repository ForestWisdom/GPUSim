"""CUTLASS-specific split-K partition helpers."""

from __future__ import annotations

from dataclasses import dataclass

from perf_model.common.constants import DTYPE_BYTES
from perf_model.common.types import GemmProblem, KernelMeta
from perf_model.common.utils import ceil_div


def _round_up(x: int, alignment: int) -> int:
    return ceil_div(x, alignment) * alignment


@dataclass(slots=True)
class CutlassKPartition:
    requested_split_k: int
    k_align: int
    gemm_k_size: int
    effective_split_k: int


def compute_cutlass_k_partition(problem: GemmProblem, kernel_meta: KernelMeta) -> CutlassKPartition:
    requested_split_k = max(problem.split_k_slices, kernel_meta.split_k_default, 1)
    dtype_bits = DTYPE_BYTES.get(kernel_meta.dtype, 2) * 8
    k_align = max(128 // dtype_bits, 1)
    gemm_k_size = _round_up(ceil_div(problem.K, requested_split_k), k_align)
    effective_split_k = ceil_div(problem.K, gemm_k_size) if gemm_k_size > 0 else 0
    return CutlassKPartition(
        requested_split_k=requested_split_k,
        k_align=k_align,
        gemm_k_size=gemm_k_size,
        effective_split_k=effective_split_k,
    )
