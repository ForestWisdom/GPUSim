"""Scheduler interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from perf_model.common.types import GemmTask, GpuSpec


class TaskScheduler(ABC):
    @abstractmethod
    def assign(self, tasks: list[GemmTask], gpu: GpuSpec) -> dict[int, list[GemmTask]]:
        raise NotImplementedError
