"""Helpers for batch Nsight Compute sweeps against analytical summaries."""

from __future__ import annotations

from pathlib import Path


def build_ncu_profile_command(
    *,
    ncu_bin: str,
    ncu_prefix: list[str],
    binary: str | Path,
    output_csv: str | Path,
    device: int,
    m: int,
    n: int,
    k: int,
    tb_m: int,
    tb_n: int,
    tb_k: int,
    split_k: int,
    swizzle: str,
    iterations: int,
    warmup: int,
    metric_names: list[str],
) -> list[str]:
    metrics = [name for name in metric_names if name]
    deduped_metrics = list(dict.fromkeys(metrics))
    if not deduped_metrics:
        raise ValueError("metric_names must include at least one metric")

    return [
        *ncu_prefix,
        ncu_bin,
        "--devices",
        str(device),
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
        "--print-metric-instances",
        "values",
        "--log-file",
        str(output_csv),
        "--metrics",
        ",".join(deduped_metrics),
        str(binary),
        "--quiet",
        "--device",
        str(device),
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--tb-m",
        str(tb_m),
        "--tb-n",
        str(tb_n),
        "--tb-k",
        str(tb_k),
        "--split-k",
        str(split_k),
        "--swizzle",
        str(swizzle),
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
    ]


def summarize_ncu_sweep_results(results: list[dict[str, object]]) -> dict[str, object]:
    total_cases = len(results)
    task_count_matched_cases = sum(1 for item in results if item["task_count_match"])
    total_tensor_ops_matched_cases = sum(1 for item in results if item["total_tensor_ops_match"])
    max_sm_supported_cases = sum(1 for item in results if item["max_sm_supported"])
    max_sm_tensor_ops_matched_cases = sum(
        1 for item in results if item.get("max_sm_tensor_ops_match") is True
    )
    fully_matched_cases = sum(
        1
        for item in results
        if item["task_count_match"]
        and item["total_tensor_ops_match"]
        and item.get("max_sm_tensor_ops_match") is True
    )
    worst_total_tensor_ops_rel_error = max(
        (float(item["total_tensor_ops_rel_error"]) for item in results),
        default=0.0,
    )
    worst_max_sm_tensor_ops_rel_error = max(
        (
            float(item["max_sm_tensor_ops_rel_error"])
            for item in results
            if item.get("max_sm_tensor_ops_rel_error") is not None
        ),
        default=0.0,
    )

    return {
        "total_cases": total_cases,
        "task_count_matched_cases": task_count_matched_cases,
        "task_count_match_rate": (task_count_matched_cases / total_cases) if total_cases else 0.0,
        "total_tensor_ops_matched_cases": total_tensor_ops_matched_cases,
        "total_tensor_ops_match_rate": (
            total_tensor_ops_matched_cases / total_cases if total_cases else 0.0
        ),
        "max_sm_supported_cases": max_sm_supported_cases,
        "max_sm_tensor_ops_matched_cases": max_sm_tensor_ops_matched_cases,
        "max_sm_tensor_ops_match_rate": (
            max_sm_tensor_ops_matched_cases / max_sm_supported_cases
            if max_sm_supported_cases
            else 0.0
        ),
        "fully_matched_cases": fully_matched_cases,
        "full_match_rate": (fully_matched_cases / total_cases) if total_cases else 0.0,
        "worst_total_tensor_ops_rel_error": worst_total_tensor_ops_rel_error,
        "worst_max_sm_tensor_ops_rel_error": worst_max_sm_tensor_ops_rel_error,
    }
