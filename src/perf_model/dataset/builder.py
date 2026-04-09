"""Build dataset rows from raw GEMM profile records."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from perf_model.common.types import GemmProblem, GpuSpec, KernelMeta
from perf_model.dataset.schema import Sample
from perf_model.pipelines.feature_pipeline import FeaturePipeline


@dataclass(slots=True)
class DatasetBuilder:
    pipeline: FeaturePipeline

    def build_samples(
        self, records: list[dict[str, int | float | str]], gpu: GpuSpec, kernel_meta: KernelMeta
    ) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for record in records:
            effective_kernel = replace(
                kernel_meta,
                swizzle=str(record.get("swizzle", kernel_meta.swizzle)),
                split_k_default=int(record.get("split_k_slices", kernel_meta.split_k_default)),
            )
            problem = GemmProblem(
                M=int(record["M"]),
                N=int(record["N"]),
                K=int(record["K"]),
                split_k_slices=int(record.get("split_k_slices", effective_kernel.split_k_default)),
            )
            features, _ = self.pipeline.run(problem, gpu, effective_kernel)
            sample = Sample(
                problem=problem,
                gpu_name=gpu.name,
                kernel_name=kernel_meta.name,
                feature_vector=features,
                latency_us=float(record["latency_us"]),
            )
            row = sample.to_row()
            row["split_k_slices"] = problem.split_k_slices
            row["swizzle"] = effective_kernel.swizzle
            rows.append(row)
        return rows

    def build_frame(
        self, records: list[dict[str, int | float | str]], gpu: GpuSpec, kernel_meta: KernelMeta
    ) -> pd.DataFrame:
        return pd.DataFrame(self.build_samples(records, gpu, kernel_meta))
