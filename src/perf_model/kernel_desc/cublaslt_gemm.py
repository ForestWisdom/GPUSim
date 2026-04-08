"""cuBLASLt heuristic normalization into the common decomposition path."""

from __future__ import annotations

from perf_model.common.types import GemmProblem, GemmTask, KernelMeta
from perf_model.kernel_desc.cutlass_gemm import CutlassGemmDecomposer


class CublasLtGemmDecomposer:
    def __init__(self) -> None:
        self._fallback = CutlassGemmDecomposer()

    def decompose(self, problem: GemmProblem, kernel_meta: KernelMeta) -> list[GemmTask]:
        return self._fallback.decompose(problem, kernel_meta)
