"""Config parsers for GEMM problem and kernel metadata."""

from __future__ import annotations

from perf_model.common.types import GemmProblem, KernelMeta
from perf_model.common.utils import load_yaml


def load_problem(record: dict[str, int]) -> GemmProblem:
    return GemmProblem(
        M=int(record["M"]),
        N=int(record["N"]),
        K=int(record["K"]),
        split_k_slices=int(record.get("split_k_slices", 1)),
    )


def load_kernel_meta(path: str) -> KernelMeta:
    config = load_yaml(path)
    return KernelMeta(
        name=str(config["name"]),
        backend=str(config["backend"]),
        pipeline=str(config["pipeline"]),
        threadblock_shape=tuple(config["threadblock_shape"]),
        warp_shape=tuple(config["warp_shape"]),
        instruction_shape=tuple(config["instruction_shape"]),
        swizzle=str(config.get("swizzle", "Identity")),
        split_k_default=int(config.get("split_k_default", 1)),
        dtype=str(config.get("dtype", "f16")),
        extra={k: v for k, v in config.items() if k not in {
            "name",
            "backend",
            "pipeline",
            "threadblock_shape",
            "warp_shape",
            "instruction_shape",
            "swizzle",
            "split_k_default",
            "dtype",
        }},
    )
