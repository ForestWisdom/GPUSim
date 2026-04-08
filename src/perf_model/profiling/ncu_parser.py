"""Nsight Compute parsing placeholder."""

from __future__ import annotations


def parse_ncu_report(path: str) -> dict[str, float]:
    return {"path_length": float(len(path))}
