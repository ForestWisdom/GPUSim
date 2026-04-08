from __future__ import annotations

from perf_model.validation.cutlass_external import compare_task_lists


def test_compare_task_lists_reports_match() -> None:
    reference_tasks = [
        {
            "tile_idx_m": 0,
            "tile_idx_n": 0,
            "tile_idx_k": 0,
            "m0": 0,
            "m1": 128,
            "n0": 0,
            "n1": 128,
            "k0": 0,
            "k1": 64,
            "m_eff": 128,
            "n_eff": 128,
            "k_eff": 64,
            "gemm_k_iterations": 2,
        }
    ]

    summary = compare_task_lists(reference_tasks, reference_tasks)

    assert summary.is_match
    assert summary.only_in_reference == []
    assert summary.only_in_model == []


def test_compare_task_lists_reports_split_k_difference() -> None:
    reference_tasks = [
        {
            "tile_idx_m": 0,
            "tile_idx_n": 0,
            "tile_idx_k": 0,
            "m0": 0,
            "m1": 128,
            "n0": 0,
            "n1": 128,
            "k0": 0,
            "k1": 24,
            "m_eff": 128,
            "n_eff": 128,
            "k_eff": 24,
            "gemm_k_iterations": 1,
        }
    ]
    model_tasks = [
        {
            "tile_idx_m": 0,
            "tile_idx_n": 0,
            "tile_idx_k": 0,
            "m0": 0,
            "m1": 128,
            "n0": 0,
            "n1": 128,
            "k0": 0,
            "k1": 32,
            "m_eff": 128,
            "n_eff": 128,
            "k_eff": 32,
            "gemm_k_iterations": 1,
        }
    ]

    summary = compare_task_lists(reference_tasks, model_tasks)

    assert not summary.is_match
    assert len(summary.only_in_reference) == 1
    assert len(summary.only_in_model) == 1
