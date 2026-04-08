"""Occupancy-aware scheduling placeholder."""

from __future__ import annotations

from perf_model.common.types import GemmTask, GpuSpec
from perf_model.scheduler.base import TaskScheduler


class ResidencyScheduler(TaskScheduler):
    def assign(self, tasks: list[GemmTask], gpu: GpuSpec) -> dict[int, list[GemmTask]]:
        raise NotImplementedError("residency-aware scheduling is not implemented yet")
