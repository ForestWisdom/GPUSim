#!/usr/bin/env python3
"""Compare CUTLASS decomposition/model summaries against Nsight Compute exports."""

from __future__ import annotations

import argparse
import json
import tempfile
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perf_model.common.types import GemmProblem, KernelMeta
from perf_model.profiling.runner import build_cutlass_gemm_bench
from perf_model.validation.ncu_compare import (
    build_comparison_payload,
    build_model_tensor_summary,
    build_ncu_tensor_summary,
    load_gpu_spec,
    probe_task_count,
)
from perf_model.validation.ncu_sweep import build_ncu_profile_command


DEFAULT_TOTAL_METRIC = "sm__inst_executed_pipe_tensor_op_hmma_v2.sum"
DEFAULT_MAX_METRIC = "sm__inst_executed_pipe_tensor_op_hmma_v2.max"
DEFAULT_METRIC_SCALE = 4096.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-config", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--tb-m", type=int, required=True)
    parser.add_argument("--tb-n", type=int, required=True)
    parser.add_argument("--tb-k", type=int, required=True)
    parser.add_argument("--warp-m", type=int, default=64)
    parser.add_argument("--warp-n", type=int, default=64)
    parser.add_argument("--warp-k", type=int, default=32)
    parser.add_argument("--inst-m", type=int, default=16)
    parser.add_argument("--inst-n", type=int, default=8)
    parser.add_argument("--inst-k", type=int, default=16)
    parser.add_argument("--split-k", type=int, default=1)
    parser.add_argument("--swizzle", default="Identity")
    parser.add_argument("--dtype", default="f16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ncu-csv")
    parser.add_argument("--total-metric", default=DEFAULT_TOTAL_METRIC)
    parser.add_argument("--metric-scale", type=float, default=DEFAULT_METRIC_SCALE)
    parser.add_argument("--max-metric", default=DEFAULT_MAX_METRIC)
    parser.add_argument("--max-metric-scale", type=float, default=DEFAULT_METRIC_SCALE)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--run-ncu", action="store_true")
    return parser.parse_args()


def _build_kernel_meta(args: argparse.Namespace) -> KernelMeta:
    return KernelMeta(
        name="cutlass_tensorop_ncu_compare",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(args.tb_m, args.tb_n, args.tb_k),
        warp_shape=(args.warp_m, args.warp_n, args.warp_k),
        instruction_shape=(args.inst_m, args.inst_n, args.inst_k),
        swizzle=args.swizzle,
        dtype=args.dtype,
    )


def _raise_ncu_permission_error(output: str) -> None:
    if "ERR_NVGPUCTRPERM" in output:
        raise SystemExit(
            "ncu failed: GPU performance counters are not accessible for this user "
            "(ERR_NVGPUCTRPERM). Enable Nsight Compute counter permissions first."
        )


def main() -> int:
    args = parse_args()
    problem = GemmProblem(M=args.m, N=args.n, K=args.k, split_k_slices=args.split_k)
    kernel_meta = _build_kernel_meta(args)
    gpu = load_gpu_spec(args.gpu_config)

    model = build_model_tensor_summary(problem, gpu, kernel_meta)

    reference_task_count: int | None = None
    if not args.skip_probe:
        reference_task_count = probe_task_count(
            PROJECT_ROOT, problem=problem, kernel_meta=kernel_meta, device=args.device
        )

    ncu_csv_path = args.ncu_csv
    if args.run_ncu:
        binary = build_cutlass_gemm_bench(PROJECT_ROOT)
        cache_dir = PROJECT_ROOT / ".cache"
        cache_dir.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="cutlass_ncu_", suffix=".csv", delete=False, dir=cache_dir
        ) as handle:
            ncu_csv_path = handle.name

        command = build_ncu_profile_command(
            ncu_bin="ncu",
            ncu_prefix=[],
            binary=binary,
            output_csv=ncu_csv_path,
            device=args.device,
            m=args.m,
            n=args.n,
            k=args.k,
            tb_m=args.tb_m,
            tb_n=args.tb_n,
            tb_k=args.tb_k,
            split_k=args.split_k,
            swizzle=args.swizzle,
            iterations=args.iterations,
            warmup=args.warmup,
            metric_names=[args.total_metric, args.max_metric],
        )
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
            _raise_ncu_permission_error(combined_output)
            raise SystemExit(combined_output or "ncu profiling failed")

    ncu = (
        build_ncu_tensor_summary(
            ncu_csv_path,
            metric_name=args.total_metric,
            metric_scale=args.metric_scale,
            max_metric_name=args.max_metric,
            max_metric_scale=args.max_metric_scale,
            model_task_count=model.task_count,
        )
        if ncu_csv_path
        else None
    )
    payload = build_comparison_payload(
        model=model,
        ncu=ncu,
        reference_task_count=reference_task_count,
    )

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
