from __future__ import annotations

from perf_model.backends.cutlass.swizzle import get_grid_shape, get_tile_offset
from perf_model.common.types import GemmProblem, KernelMeta
from perf_model.kernel_desc.cutlass_gemm import CutlassGemmDecomposer


def _make_kernel(swizzle: str) -> KernelMeta:
    return KernelMeta(
        name="cutlass_tensorop",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(128, 128, 32),
        warp_shape=(64, 64, 32),
        instruction_shape=(16, 8, 16),
        swizzle=swizzle,
    )


def test_identity2_grid_shape_matches_cutlass_formula() -> None:
    tiled_shape = (3, 5, 2)

    assert get_grid_shape("Identity2", tiled_shape) == (6, 3, 2)


def test_identity2_tile_offsets_match_cutlass_formula() -> None:
    tiled_shape = (3, 5, 2)

    offsets = [
        get_tile_offset("Identity2", block_x, block_y, 0, tiled_shape)
        for block_y in range(3)
        for block_x in range(6)
    ]

    valid_offsets = [offset for offset in offsets if offset[0] < tiled_shape[0] and offset[1] < tiled_shape[1]]
    assert valid_offsets[:8] == [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
        (2, 0, 0),
        (2, 1, 0),
        (0, 2, 0),
        (0, 3, 0),
    ]


def test_identity4_decomposer_preserves_logical_coverage() -> None:
    problem = GemmProblem(M=256, N=640, K=64, split_k_slices=1)
    identity_tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel("Identity"))
    identity4_tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel("Identity4"))

    assert len(identity_tasks) == len(identity4_tasks)
    assert {
        (task.tile_idx_m, task.tile_idx_n, task.tile_idx_k, task.m0, task.n0, task.k0)
        for task in identity_tasks
    } == {
        (task.tile_idx_m, task.tile_idx_n, task.tile_idx_k, task.m0, task.n0, task.k0)
        for task in identity4_tasks
    }

    assert [
        (task.tile_idx_m, task.tile_idx_n, task.tile_idx_k) for task in identity_tasks
    ] != [
        (task.tile_idx_m, task.tile_idx_n, task.tile_idx_k) for task in identity4_tasks
    ]
