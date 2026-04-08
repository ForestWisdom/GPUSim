#!/usr/bin/env python3
"""Thin CLI wrapper around the dataset builder."""

from __future__ import annotations

import argparse

import pandas as pd

from perf_model.common.types import GpuSpec
from perf_model.common.utils import load_yaml
from perf_model.dataset.builder import DatasetBuilder
from perf_model.kernel_desc.parser import load_kernel_meta
from perf_model.pipelines.feature_pipeline import build_default_feature_pipeline


def load_gpu_spec(path: str) -> GpuSpec:
    config = load_yaml(path)
    return GpuSpec(
        name=str(config["name"]),
        num_sms=int(config["num_sms"]),
        tensor_throughput_per_sm=float(config["tensor_throughput_per_sm"]),
        simt_throughput_per_sm=float(config["simt_throughput_per_sm"]),
        dram_bw_gbps=float(config["dram_bw_gbps"]),
        l2_bw_gbps=float(config["l2_bw_gbps"]),
        smem_bw_gbps_per_sm=float(config["smem_bw_gbps_per_sm"]),
        clock_mhz=float(config["clock_mhz"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with M,N,K,latency_us")
    parser.add_argument("--output", required=True, help="Output processed CSV path")
    parser.add_argument("--gpu", required=True, help="GPU yaml config")
    parser.add_argument("--kernel", required=True, help="Kernel yaml config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu = load_gpu_spec(args.gpu)
    kernel_meta = load_kernel_meta(args.kernel)
    frame = pd.read_csv(args.input)
    records = frame.to_dict(orient="records")
    pipeline = build_default_feature_pipeline(kernel_meta)
    builder = DatasetBuilder(pipeline)
    dataset = builder.build_frame(records, gpu, kernel_meta)
    dataset.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
