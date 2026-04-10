#!/usr/bin/env python3
"""Inspect one GEMM problem through the feature pipeline."""

from __future__ import annotations

import argparse

from perf_model.common.types import GemmProblem, GpuSpec
from perf_model.common.utils import load_yaml
from perf_model.kernel_desc.parser import load_kernel_meta
from perf_model.pipelines.feature_pipeline import build_default_feature_pipeline


def load_gpu_spec(path: str) -> GpuSpec:
    config = load_yaml(path)
    return GpuSpec(
        name=str(config["name"]),
        num_sms=int(config["num_sms"]),
        tensor_throughput_per_sm=float(config["tensor_throughput_per_sm"]),
        simt_throughput_per_sm=float(config["simt_throughput_per_sm"]),
        dram_bw_bytes_per_cycle=float(config["dram_bw_bytes_per_cycle"]),
        l2_bw_bytes_per_cycle=float(config["l2_bw_bytes_per_cycle"]),
        smem_bw_bytes_per_cycle_per_sm=float(config["smem_bw_bytes_per_cycle_per_sm"]),
        clock_mhz=float(config["clock_mhz"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--split-k", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu = load_gpu_spec(args.gpu)
    kernel_meta = load_kernel_meta(args.kernel)
    pipeline = build_default_feature_pipeline(kernel_meta)
    features, debug = pipeline.run(
        GemmProblem(M=args.m, N=args.n, K=args.k, split_k_slices=args.split_k),
        gpu,
        kernel_meta,
    )
    print(f"num_tasks={len(debug['tasks'])}")
    print(f"feature_dim={len(features)}")
    print(f"max_busy_cycles={max(sm.estimated_busy_cycles for sm in debug['sm_features']):.6f}")


if __name__ == "__main__":
    main()
