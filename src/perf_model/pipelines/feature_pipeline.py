"""Compose decomposition, scheduling, and feature building."""

from __future__ import annotations

from dataclasses import dataclass

from perf_model.common.types import GemmProblem, GpuSpec, KernelMeta, SmFeatures, TaskFeatures
from perf_model.features.feature_vector import build_feature_vector
from perf_model.kernel_desc.cutlass_gemm import CutlassGemmDecomposer
from perf_model.scheduler.round_robin import RoundRobinScheduler


@dataclass(slots=True)
class FeaturePipeline:
    decomposer: object
    scheduler: object
    feature_builder: object

    def run(
        self, problem: GemmProblem, gpu: GpuSpec, kernel_meta: KernelMeta
    ) -> tuple[list[float], dict[str, object]]:
        tasks = self.decomposer.decompose(problem, kernel_meta)
        assignments = self.scheduler.assign(tasks, gpu)

        sm_features: list[SmFeatures] = []
        task_features_by_sm: dict[int, list[TaskFeatures]] = {}
        for sm_id, sm_tasks in assignments.items():
            task_features = [
                self.feature_builder.build_task_features(task, sm_id, gpu, kernel_meta)
                for task in sm_tasks
            ]
            task_features_by_sm[sm_id] = task_features
            sm_features.append(self.feature_builder.aggregate_sm_features(sm_id, task_features))

        aggregated = self.feature_builder.aggregate_gpu_features(sm_features)
        return build_feature_vector(problem, gpu, kernel_meta, aggregated), {
            "tasks": tasks,
            "assignments": assignments,
            "task_features_by_sm": task_features_by_sm,
            "sm_features": sm_features,
        }


def build_default_feature_pipeline(kernel_meta: KernelMeta) -> FeaturePipeline:
    if kernel_meta.pipeline == "simt":
        from perf_model.features.gemm_simt import SimtFeatureBuilder

        builder = SimtFeatureBuilder()
    else:
        from perf_model.features.gemm_tensor_core import TensorCoreFeatureBuilder

        builder = TensorCoreFeatureBuilder()

    return FeaturePipeline(
        decomposer=CutlassGemmDecomposer(),
        scheduler=RoundRobinScheduler(),
        feature_builder=builder,
    )
