"""CUTLASS-style GEMM CTA decomposition."""
from __future__ import annotations

from perf_model.backends.cutlass.partition import compute_cutlass_k_partition
from perf_model.backends.cutlass.swizzle import (
    get_grid_shape,
    get_tile_offset,
    get_tiled_shape,
)
from perf_model.common.types import GemmProblem, GemmTask, KernelMeta
from perf_model.common.utils import ceil_div
from perf_model.kernel_desc.base import KernelDecomposer


class CutlassGemmDecomposer(KernelDecomposer):
    """Approximate CUTLASS 2.x GEMM decomposition at CTA granularity."""

    def decompose(self, problem: GemmProblem, kernel_meta: KernelMeta) -> list[GemmTask]:
        tb_m, tb_n, tb_k = kernel_meta.threadblock_shape
        partition = compute_cutlass_k_partition(problem, kernel_meta)

        tiled_shape = get_tiled_shape(
            problem.M,
            problem.N,
            problem.K,
            tb_m,
            tb_n,
            tb_k,
            partition.effective_split_k,
        )
        grid_x, grid_y, grid_z = get_grid_shape(kernel_meta.swizzle, tiled_shape)

        tasks: list[GemmTask] = []
        task_idx = 0

        for block_z in range(grid_z):
            for block_y in range(grid_y):
                for block_x in range(grid_x):
                    tile_idx_m, tile_idx_n, tile_idx_k = get_tile_offset(
                        kernel_meta.swizzle,
                        block_x,
                        block_y,
                        block_z,
                        tiled_shape,
                    )

                    m0 = tile_idx_m * tb_m
                    m1 = min(problem.M, m0 + tb_m)
                    n0 = tile_idx_n * tb_n
                    n1 = min(problem.N, n0 + tb_n)

                    k0 = tile_idx_k * partition.gemm_k_size
                    k1 = min(problem.K, k0 + partition.gemm_k_size)

                    if k0 >= k1 or m0 >= m1 or n0 >= n1:
                        continue

                    m_eff = m1 - m0
                    n_eff = n1 - n0
                    k_eff = k1 - k0

                    task = GemmTask(
                        tile_m=tb_m,
                        tile_n=tb_n,
                        tile_k=tb_k,
                        m0=m0,
                        m1=m1,
                        n0=n0,
                        n1=n1,
                        k0=k0,
                        k1=k1,
                        m_eff=m_eff,
                        n_eff=n_eff,
                        k_eff=k_eff,
                        gemm_k_iterations=ceil_div(k_eff, tb_k),
                        task_idx=task_idx,
                        tile_idx_m=tile_idx_m,
                        tile_idx_n=tile_idx_n,
                        tile_idx_k=tile_idx_k,
                        is_edge_m=(m_eff < tb_m),
                        is_edge_n=(n_eff < tb_n),
                        is_edge_k=(k_eff < partition.gemm_k_size),
                    )
                    tasks.append(task)
                    task_idx += 1

        return tasks
