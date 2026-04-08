"""Round-robin task assignment across SMs."""

from __future__ import annotations

from perf_model.common.types import GemmTask, GpuSpec
from perf_model.scheduler.base import TaskScheduler


class RoundRobinScheduler(TaskScheduler):
    def assign(self, tasks: list[GemmTask], gpu: GpuSpec) -> dict[int, list[GemmTask]]:
        assignments = {sm_id: [] for sm_id in range(gpu.num_sms)}
        for index, task in enumerate(tasks):
            assignments[index % gpu.num_sms].append(task)
        return assignments
