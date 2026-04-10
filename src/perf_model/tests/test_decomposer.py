from __future__ import annotations

from collections import defaultdict

from perf_model.common.types import GemmProblem, KernelMeta
from perf_model.kernel_desc.cutlass_gemm import CutlassGemmDecomposer


def _make_kernel(swizzle: str = "Identity") -> KernelMeta:
    return KernelMeta(
        name="cutlass_tensorop",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(128, 128, 32),
        warp_shape=(64, 64, 32),
        instruction_shape=(16, 8, 16),
        swizzle=swizzle,
    )


def _coverage_by_output_tile(tasks: list) -> dict[tuple[int, int], list]:
    buckets: dict[tuple[int, int], list] = defaultdict(list)
    for task in tasks:
        buckets[(task.m0, task.n0)].append(task)
    return buckets


def test_case_a_regular_tiles_have_expected_task_count_and_ranges() -> None:
    problem = GemmProblem(M=256, N=256, K=64, split_k_slices=1)
    tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel())

    assert len(tasks) == 4
    assert {(task.m0, task.m1, task.n0, task.n1) for task in tasks} == {
        (0, 128, 0, 128),
        (0, 128, 128, 256),
        (128, 256, 0, 128),
        (128, 256, 128, 256),
    }
    assert all(task.k0 == 0 and task.k1 == 64 for task in tasks)
    assert all(task.gemm_k_iterations == 2 for task in tasks)


def test_case_b_edge_tiles_are_clipped_without_empty_tasks() -> None:
    problem = GemmProblem(M=250, N=250, K=64, split_k_slices=1)
    tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel())

    assert len(tasks) == 4
    assert all(task.m_eff > 0 and task.n_eff > 0 and task.k_eff > 0 for task in tasks)
    assert max(task.m_eff for task in tasks) == 128
    assert max(task.n_eff for task in tasks) == 128
    assert any(task.m_eff == 122 for task in tasks)
    assert any(task.n_eff == 122 for task in tasks)
    assert any(task.is_edge_m for task in tasks)
    assert any(task.is_edge_n for task in tasks)


def test_case_c_split_k_partitions_follow_cutlass_k_alignment() -> None:
    problem = GemmProblem(M=256, N=256, K=96, split_k_slices=2)
    tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel())

    assert len(tasks) == 8
    coverage = _coverage_by_output_tile(tasks)
    assert len(coverage) == 4
    assert all(len(tile_tasks) == 2 for tile_tasks in coverage.values())
    for tile_tasks in coverage.values():
        ordered = sorted(tile_tasks, key=lambda task: task.k0)
        assert [(task.k0, task.k1, task.gemm_k_iterations) for task in ordered] == [
            (0, 48, 2),
            (48, 96, 2),
        ]


def test_case_d_small_edge_problem_keeps_valid_k_partitions() -> None:
    problem = GemmProblem(M=130, N=129, K=33, split_k_slices=2)
    tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel())

    assert len(tasks) == 8
    assert all(task.m_eff > 0 and task.n_eff > 0 and task.k_eff > 0 for task in tasks)
    assert all(task.m_eff <= 128 and task.n_eff <= 128 for task in tasks)
    assert all(task.k_eff <= 32 for task in tasks)

    coverage = _coverage_by_output_tile(tasks)
    assert len(coverage) == 4
    for tile_tasks in coverage.values():
        ordered = sorted(tile_tasks, key=lambda task: task.k0)
        assert [(task.k0, task.k1, task.k_eff) for task in ordered] == [
            (0, 24, 24),
            (24, 33, 9),
        ]


def test_split_k_partitions_follow_cutlass_k_alignment_for_f16() -> None:
    problem = GemmProblem(M=128, N=128, K=33, split_k_slices=2)
    kernel = _make_kernel()

    tasks = CutlassGemmDecomposer().decompose(problem, kernel)

    assert len(tasks) == 2
    ordered = sorted(tasks, key=lambda task: task.k0)
    assert [(task.k0, task.k1, task.k_eff, task.gemm_k_iterations) for task in ordered] == [
        (0, 24, 24, 1),
        (24, 33, 9, 1),
    ]


def test_swizzle_changes_order_but_not_logical_coverage() -> None:
    problem = GemmProblem(M=256, N=384, K=64, split_k_slices=2)
    identity_tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel("Identity"))
    horizontal_tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel("Horizontal"))

    assert len(identity_tasks) == len(horizontal_tasks) == 12

    identity_triplets = {
        (task.tile_idx_m, task.tile_idx_n, task.tile_idx_k, task.m0, task.n0, task.k0)
        for task in identity_tasks
    }
    horizontal_triplets = {
        (task.tile_idx_m, task.tile_idx_n, task.tile_idx_k, task.m0, task.n0, task.k0)
        for task in horizontal_tasks
    }
    assert identity_triplets == horizontal_triplets

    identity_order = [(task.tile_idx_m, task.tile_idx_n, task.tile_idx_k) for task in identity_tasks]
    horizontal_order = [(task.tile_idx_m, task.tile_idx_n, task.tile_idx_k) for task in horizontal_tasks]
    assert identity_order != horizontal_order


def test_split_k_greater_than_one_scales_task_count_by_k_partitions() -> None:
    problem = GemmProblem(M=256, N=256, K=128, split_k_slices=4)
    tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel())

    assert len(tasks) == 16
    coverage = _coverage_by_output_tile(tasks)
    assert len(coverage) == 4
    assert all(len(tile_tasks) == 4 for tile_tasks in coverage.values())
    for tile_tasks in coverage.values():
        ordered = sorted(tile_tasks, key=lambda task: task.k0)
        assert [(task.k0, task.k1) for task in ordered] == [
            (0, 32),
            (32, 64),
            (64, 96),
            (96, 128),
        ]
        assert [task.tile_idx_k for task in ordered] == [0, 1, 2, 3]


def test_identity_n_swizzles_preserve_logical_coverage() -> None:
    problem = GemmProblem(M=256, N=640, K=64, split_k_slices=1)
    identity_tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel("Identity"))
    identity2_tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel("Identity2"))
    identity4_tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel("Identity4"))

    def _signature(tasks: list) -> set[tuple[int, int, int, int, int, int]]:
        return {
            (task.tile_idx_m, task.tile_idx_n, task.m0, task.m1, task.n0, task.n1)
            for task in tasks
        }

    assert len(identity_tasks) == len(identity2_tasks) == len(identity4_tasks) == 10
    assert _signature(identity_tasks) == _signature(identity2_tasks) == _signature(identity4_tasks)
    assert [(task.tile_idx_m, task.tile_idx_n) for task in identity_tasks] != [
        (task.tile_idx_m, task.tile_idx_n) for task in identity2_tasks
    ]


def test_swizzles_preserve_task_count_and_output_coverage_under_split_k() -> None:
    problem = GemmProblem(M=256, N=384, K=65, split_k_slices=2)
    decomposer = CutlassGemmDecomposer()
    task_sets = {
        swizzle: decomposer.decompose(problem, _make_kernel(swizzle))
        for swizzle in ("Identity", "Identity2", "Identity4", "Horizontal")
    }

    expected_task_count = 12
    expected_signature = {
        (task.tile_idx_m, task.tile_idx_n, task.tile_idx_k, task.m0, task.m1, task.n0, task.n1, task.k0, task.k1)
        for task in task_sets["Identity"]
    }

    for tasks in task_sets.values():
        assert len(tasks) == expected_task_count
        assert {
            (task.tile_idx_m, task.tile_idx_n, task.tile_idx_k, task.m0, task.m1, task.n0, task.n1, task.k0, task.k1)
            for task in tasks
        } == expected_signature


def test_split_k_output_tiles_map_to_multiple_distinct_tile_k_tasks() -> None:
    problem = GemmProblem(M=256, N=256, K=65, split_k_slices=2)
    tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel())

    coverage = _coverage_by_output_tile(tasks)
    assert len(coverage) == 4
    for tile_tasks in coverage.values():
        ordered = sorted(tile_tasks, key=lambda task: task.tile_idx_k)
        assert [task.tile_idx_k for task in ordered] == [0, 1]
        assert [(task.k0, task.k1, task.gemm_k_iterations) for task in ordered] == [
            (0, 40, 2),
            (40, 65, 1),
        ]


def test_k_not_multiple_of_tb_k_keeps_tail_iteration_in_last_task() -> None:
    problem = GemmProblem(M=128, N=128, K=65, split_k_slices=1)
    tasks = CutlassGemmDecomposer().decompose(problem, _make_kernel())

    assert len(tasks) == 1
    task = tasks[0]
    assert (task.k0, task.k1) == (0, 65)
    assert task.k_eff == 65
    assert task.gemm_k_iterations == 3
    assert task.is_edge_k
