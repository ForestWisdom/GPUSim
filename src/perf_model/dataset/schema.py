"""Dataset record schemas."""

from __future__ import annotations

from dataclasses import dataclass

from perf_model.common.types import GemmProblem


@dataclass(slots=True)
class Sample:
    problem: GemmProblem
    gpu_name: str
    kernel_name: str
    feature_vector: list[float]
    latency_us: float

    def to_row(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "M": self.problem.M,
            "N": self.problem.N,
            "K": self.problem.K,
            "split_k_slices": self.problem.split_k_slices,
            "gpu_name": self.gpu_name,
            "kernel_name": self.kernel_name,
            "latency_us": self.latency_us,
        }
        for index, value in enumerate(self.feature_vector):
            row[f"f_{index}"] = value
        return row
