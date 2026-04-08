"""Feature builder interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from perf_model.common.types import GemmTask, GpuSpec, KernelMeta, SmFeatures, TaskFeatures


class FeatureBuilder(ABC):
    @abstractmethod
    def build_task_features(
        self, task: GemmTask, sm_id: int, gpu: GpuSpec, kernel_meta: KernelMeta
    ) -> TaskFeatures:
        raise NotImplementedError

    @abstractmethod
    def aggregate_sm_features(self, sm_id: int, task_features: list[TaskFeatures]) -> SmFeatures:
        raise NotImplementedError

    @abstractmethod
    def aggregate_gpu_features(self, sm_features: list[SmFeatures]) -> list[float]:
        raise NotImplementedError
