"""Interfaces for mapping kernels into task lists."""

from __future__ import annotations

from abc import ABC, abstractmethod

from perf_model.common.types import GemmProblem, GemmTask, KernelMeta


class KernelDecomposer(ABC):
    @abstractmethod
    def decompose(self, problem: GemmProblem, kernel_meta: KernelMeta) -> list[GemmTask]:
        raise NotImplementedError
