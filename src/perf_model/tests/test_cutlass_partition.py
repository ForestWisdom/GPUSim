from __future__ import annotations

from perf_model.backends.cutlass.partition import compute_cutlass_k_partition
from perf_model.common.types import GemmProblem, KernelMeta


def test_compute_cutlass_k_partition_for_f16_split_k() -> None:
    problem = GemmProblem(M=128, N=128, K=33, split_k_slices=2)
    kernel = KernelMeta(
        name="cutlass_tensorop",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(128, 128, 32),
        warp_shape=(64, 64, 32),
        instruction_shape=(16, 8, 16),
        dtype="f16",
    )

    partition = compute_cutlass_k_partition(problem, kernel)

    assert partition.k_align == 8
    assert partition.gemm_k_size == 24
    assert partition.effective_split_k == 2
