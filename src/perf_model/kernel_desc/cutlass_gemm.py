"""CUTLASS-style GEMM CTA decomposition."""

from __future__ import annotations

from perf_model.common.types import GemmProblem, GemmTask, KernelMeta
from perf_model.common.utils import ceil_div
from perf_model.kernel_desc.base import KernelDecomposer


class CutlassGemmDecomposer(KernelDecomposer):
    def decompose(self, problem: GemmProblem, kernel_meta: KernelMeta) -> list[GemmTask]:
        tb_m, tb_n, tb_k = kernel_meta.threadblock_shape
        split_k = max(problem.split_k_slices, kernel_meta.split_k_default, 1)
        k_slice = ceil_div(problem.K, split_k)

        tasks: list[GemmTask] = []
        task_idx = 0
        for tile_idx_m in range(ceil_div(problem.M, tb_m)):
            m0 = tile_idx_m * tb_m
            m1 = min(problem.M, m0 + tb_m)
            for tile_idx_n in range(ceil_div(problem.N, tb_n)):
                n0 = tile_idx_n * tb_n
                n1 = min(problem.N, n0 + tb_n)
                for tile_idx_k in range(split_k):
                    k0 = tile_idx_k * k_slice
                    k1 = min(problem.K, k0 + k_slice)
                    if k0 >= k1:
                        continue
                    k_eff = k1 - k0
                    tasks.append(
                        GemmTask(
                            tile_m=tb_m,
                            tile_n=tb_n,
                            tile_k=tb_k,
                            m0=m0,
                            m1=m1,
                            n0=n0,
                            n1=n1,
                            k0=k0,
                            k1=k1,
                            m_eff=m1 - m0,
                            n_eff=n1 - n0,
                            k_eff=k_eff,
                            gemm_k_iterations=ceil_div(k_eff, tb_k),
                            task_idx=task_idx,
                            tile_idx_m=tile_idx_m,
                            tile_idx_n=tile_idx_n,
                            tile_idx_k=tile_idx_k,
                        )
                    )
                    task_idx += 1
        return tasks
