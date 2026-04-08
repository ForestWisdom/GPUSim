"""Decode heuristic metadata into common kernel metadata fields."""

from __future__ import annotations

from perf_model.backends.cublaslt.heuristic import HeuristicResult


def decode_algo(result: HeuristicResult) -> dict[str, int | str | float]:
    return {
        "tile_id": result.tile_id,
        "split_k": result.split_k,
        "swizzle": result.swizzle,
        "workspace_bytes": result.workspace_bytes,
        "waves_count": result.waves_count,
    }
