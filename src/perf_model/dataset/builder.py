"""Build dataset rows from raw GEMM profile records."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from perf_model.common.types import GemmProblem, GpuSpec, KernelMeta
from perf_model.dataset.schema import Sample
from perf_model.pipelines.feature_pipeline import FeaturePipeline


@dataclass(slots=True)
class DatasetBuilder:
    pipeline: FeaturePipeline

    def build_samples(
        self, records: list[dict[str, int | float]], gpu: GpuSpec, kernel_meta: KernelMeta
    ) -> list[Sample]:
        samples: list[Sample] = []
        for record in records:
            problem = GemmProblem(
                M=int(record["M"]),
                N=int(record["N"]),
                K=int(record["K"]),
                split_k_slices=int(record.get("split_k_slices", 1)),
            )
            features, _ = self.pipeline.run(problem, gpu, kernel_meta)
            samples.append(
                Sample(
                    problem=problem,
                    gpu_name=gpu.name,
                    kernel_name=kernel_meta.name,
                    feature_vector=features,
                    latency_us=float(record["latency_us"]),
                )
            )
        return samples

    def build_frame(
        self, records: list[dict[str, int | float]], gpu: GpuSpec, kernel_meta: KernelMeta
    ) -> pd.DataFrame:
        samples = self.build_samples(records, gpu, kernel_meta)
        return pd.DataFrame([sample.to_row() for sample in samples])
