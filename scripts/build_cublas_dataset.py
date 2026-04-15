#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perf_model.backends.cublaslt.kernel_name_parser import parse_cublas_kernel_name
from perf_model.backends.cublaslt.normalize import build_cublas_kernel_meta
from perf_model.common.types import GemmProblem
from perf_model.common.utils import load_yaml
from perf_model.pipelines.feature_pipeline import build_default_feature_pipeline
from perf_model.common.types import GpuSpec


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


def build_cublas_dataset_frame(frame: pd.DataFrame, *, gpu_yaml: str) -> pd.DataFrame:
    gpu = load_gpu_spec(gpu_yaml)
    rows: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        parsed = parse_cublas_kernel_name(str(record["kernel_name"]))
        kernel_meta = build_cublas_kernel_meta(parsed, dtype=str(record["dtype"]))
        pipeline = build_default_feature_pipeline(kernel_meta)
        problem = GemmProblem(M=int(record["M"]), N=int(record["N"]), K=int(record["K"]))
        feature_vector, _context = pipeline.run(problem, gpu, kernel_meta)
        row = dict(record)
        for idx, value in enumerate(feature_vector):
            row[f"f_{idx}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    frame = pd.read_csv(args.input)
    dataset = build_cublas_dataset_frame(frame, gpu_yaml=args.gpu)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)


if __name__ == "__main__":
    main()
