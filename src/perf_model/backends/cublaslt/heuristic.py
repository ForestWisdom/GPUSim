"""Heuristic collection placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HeuristicResult:
    tile_id: int
    split_k: int
    swizzle: str
    workspace_bytes: int = 0
    waves_count: float = 1.0


def collect_heuristic_stub() -> list[HeuristicResult]:
    return []
