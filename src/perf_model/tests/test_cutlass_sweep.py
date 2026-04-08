from __future__ import annotations

from perf_model.validation.cutlass_external import summarize_sweep_results


def test_summarize_sweep_results_counts_matches_and_mismatches() -> None:
    summary = summarize_sweep_results(
        [
            {"match": True, "device": 0, "swizzle": "Identity", "dtype": "f16"},
            {"match": False, "device": 0, "swizzle": "Identity", "dtype": "f16"},
            {"match": True, "device": 7, "swizzle": "Horizontal", "dtype": "f16"},
        ]
    )

    assert summary["total_cases"] == 3
    assert summary["matched_cases"] == 2
    assert summary["mismatched_cases"] == 1
    assert summary["match_rate"] == 2 / 3


def test_summarize_sweep_results_groups_mismatches_by_signature() -> None:
    summary = summarize_sweep_results(
        [
            {
                "match": False,
                "device": 7,
                "swizzle": "Identity",
                "dtype": "f16",
                "tb_m": 128,
                "tb_n": 128,
                "tb_k": 32,
                "split_k": 2,
            },
            {
                "match": False,
                "device": 5,
                "swizzle": "Identity",
                "dtype": "f16",
                "tb_m": 128,
                "tb_n": 128,
                "tb_k": 32,
                "split_k": 2,
            },
        ]
    )

    assert len(summary["mismatch_groups"]) == 1
    assert summary["mismatch_groups"][0]["count"] == 2
    assert summary["mismatch_groups"][0]["swizzle"] == "Identity"
