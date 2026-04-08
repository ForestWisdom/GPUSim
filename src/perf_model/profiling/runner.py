"""Benchmark runner placeholder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProfileResult:
    latency_us: float
    repeats: int


def run_benchmark_stub(repeats: int = 100) -> ProfileResult:
    return ProfileResult(latency_us=0.0, repeats=repeats)
