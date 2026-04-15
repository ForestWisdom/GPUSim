"""Empirical decomposition helpers for profiled cuBLAS GEMM calls."""

from __future__ import annotations

import re

_TILE_PATTERN = re.compile(r"(?<!\d)\d+x\d+(?!\d)")


def kernel_family_name(kernel_name: str) -> str:
    return _TILE_PATTERN.sub("tile", kernel_name)


def summarize_gemm_call(rows: list[dict[str, int | float | str | bool]]) -> dict[str, int | str | bool]:
    main_rows = [row for row in rows if not bool(row["is_reduction_kernel"])]
    if not main_rows:
        raise ValueError("no main kernel rows found")

    main_row = min(main_rows, key=lambda row: int(row["kernel_index"]))
    task_count = int(main_row["grid_x"]) * int(main_row["grid_y"]) * int(main_row["grid_z"])
    return {
        "main_kernel_family": kernel_family_name(str(main_row["kernel_name"])),
        "main_kernel_name": str(main_row["kernel_name"]),
        "total_kernel_count": len(rows),
        "has_reduction_kernel": any(bool(row["is_reduction_kernel"]) for row in rows),
        "main_kernel_grid_x": int(main_row["grid_x"]),
        "main_kernel_grid_y": int(main_row["grid_y"]),
        "main_kernel_grid_z": int(main_row["grid_z"]),
        "main_kernel_task_count": task_count,
    }
