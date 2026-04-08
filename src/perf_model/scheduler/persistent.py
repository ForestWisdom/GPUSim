"""Persistent kernel scheduling placeholder."""

from __future__ import annotations

from perf_model.common.types import GemmTask, GpuSpec
from perf_model.scheduler.base import TaskScheduler


class PersistentScheduler(TaskScheduler):
    def assign(self, tasks: list[GemmTask], gpu: GpuSpec) -> dict[int, list[GemmTask]]:
        raise NotImplementedError("persistent kernel scheduling is not implemented yet")
