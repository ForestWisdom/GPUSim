#!/usr/bin/env python3
"""Run a batch Nsight Compute sweep against CUTLASS analytical summaries."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perf_model.common.types import GemmProblem, KernelMeta
from perf_model.profiling.runner import build_cutlass_gemm_bench
from perf_model.validation.cutlass_external import generate_cases
from perf_model.validation.ncu_compare import (
    build_model_tensor_summary,
    build_ncu_tensor_summary,
    load_gpu_spec,
    probe_task_count,
)
from perf_model.validation.ncu_sweep import build_ncu_profile_command, summarize_ncu_sweep_results


DEFAULT_TOTAL_METRIC = "sm__inst_executed_pipe_tensor_op_hmma_v2.sum"
DEFAULT_MAX_METRIC = "sm__inst_executed_pipe_tensor_op_hmma_v2.max"
DEFAULT_METRIC_SCALE = 4096.0


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_tb_shapes(value: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for item in _parse_str_list(value):
        m_str, n_str, k_str = item.lower().split("x")
        shapes.append((int(m_str), int(n_str), int(k_str)))
    return shapes


def _build_kernel_meta(
    *,
    tb_m: int,
    tb_n: int,
    tb_k: int,
    swizzle: str,
    dtype: str,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    inst_m: int,
    inst_n: int,
    inst_k: int,
) -> KernelMeta:
    return KernelMeta(
        name="cutlass_tensorop_ncu_sweep",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(tb_m, tb_n, tb_k),
        warp_shape=(warp_m, warp_n, warp_k),
        instruction_shape=(inst_m, inst_n, inst_k),
        swizzle=swizzle,
        dtype=dtype,
    )


def _raise_ncu_permission_error(output: str) -> None:
    if "ERR_NVGPUCTRPERM" in output:
        raise SystemExit(
            "ncu failed: GPU performance counters are not accessible for this user "
            "(ERR_NVGPUCTRPERM). Enable Nsight Compute counter permissions first."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-config", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m-values", default="128,256")
    parser.add_argument("--n-values", default="128,256")
    parser.add_argument("--k-values", default="128")
    parser.add_argument("--tb-shapes", default="128x128x32")
    parser.add_argument("--split-k-values", default="1")
    parser.add_argument("--swizzles", default="Identity")
    parser.add_argument("--dtypes", default="f16")
    parser.add_argument("--warp-m", type=int, default=64)
    parser.add_argument("--warp-n", type=int, default=64)
    parser.add_argument("--warp-k", type=int, default=32)
    parser.add_argument("--inst-m", type=int, default=16)
    parser.add_argument("--inst-n", type=int, default=8)
    parser.add_argument("--inst-k", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--total-metric", default=DEFAULT_TOTAL_METRIC)
    parser.add_argument("--metric-scale", type=float, default=DEFAULT_METRIC_SCALE)
    parser.add_argument("--max-metric", default=DEFAULT_MAX_METRIC)
    parser.add_argument("--max-metric-scale", type=float, default=DEFAULT_METRIC_SCALE)
    parser.add_argument("--max-relative-error", type=float, default=1e-9)
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--ncu-prefix", default="")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--stop-on-mismatch", action="store_true")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu = load_gpu_spec(args.gpu_config)
    binary = build_cutlass_gemm_bench(PROJECT_ROOT)
    cases = generate_cases(
        m_values=_parse_int_list(args.m_values),
        n_values=_parse_int_list(args.n_values),
        k_values=_parse_int_list(args.k_values),
        tb_shapes=_parse_tb_shapes(args.tb_shapes),
        split_k_values=_parse_int_list(args.split_k_values),
        swizzles=_parse_str_list(args.swizzles),
        dtypes=_parse_str_list(args.dtypes),
    )
    if args.case_limit is not None:
        cases = cases[: args.case_limit]

    ncu_prefix = shlex.split(args.ncu_prefix)
    cache_dir = PROJECT_ROOT / ".cache" / "ncu_sweep"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={args.device} total_cases={len(cases)} gpu={gpu.name}")

    results: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        problem = GemmProblem(M=case.m, N=case.n, K=case.k, split_k_slices=case.split_k)
        kernel_meta = _build_kernel_meta(
            tb_m=case.tb_m,
            tb_n=case.tb_n,
            tb_k=case.tb_k,
            swizzle=case.swizzle,
            dtype=case.dtype,
            warp_m=args.warp_m,
            warp_n=args.warp_n,
            warp_k=args.warp_k,
            inst_m=args.inst_m,
            inst_n=args.inst_n,
            inst_k=args.inst_k,
        )
        model = build_model_tensor_summary(problem, gpu, kernel_meta)
        reference_task_count = (
            None
            if args.skip_probe
            else probe_task_count(
                PROJECT_ROOT,
                problem=problem,
                kernel_meta=kernel_meta,
                device=args.device,
            )
        )

        with tempfile.NamedTemporaryFile(
            prefix="cutlass_ncu_",
            suffix=".csv",
            delete=False,
            dir=cache_dir,
        ) as handle:
            ncu_csv_path = Path(handle.name)

        command = build_ncu_profile_command(
            ncu_bin=args.ncu_bin,
            ncu_prefix=ncu_prefix,
            binary=binary,
            output_csv=ncu_csv_path,
            device=args.device,
            m=case.m,
            n=case.n,
            k=case.k,
            tb_m=case.tb_m,
            tb_n=case.tb_n,
            tb_k=case.tb_k,
            split_k=case.split_k,
            swizzle=case.swizzle,
            iterations=args.iterations,
            warmup=args.warmup,
            metric_names=[args.total_metric, args.max_metric],
        )
        profile = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if profile.returncode != 0:
            combined_output = "\n".join(
                part for part in (profile.stdout, profile.stderr) if part
            )
            _raise_ncu_permission_error(combined_output)
            raise SystemExit(combined_output or "ncu profiling failed")

        ncu = build_ncu_tensor_summary(
            ncu_csv_path,
            metric_name=args.total_metric,
            metric_scale=args.metric_scale,
            max_metric_name=args.max_metric,
            max_metric_scale=args.max_metric_scale,
            model_task_count=model.task_count,
        )
        task_count_match = (
            reference_task_count == model.task_count
            if reference_task_count is not None
            else True
        )
        total_rel_error = abs(model.total_tensor_ops - ncu.total_tensor_ops) / max(
            abs(model.total_tensor_ops),
            abs(ncu.total_tensor_ops),
            1.0,
        )
        total_match = total_rel_error <= args.max_relative_error
        max_rel_error = None
        max_match = None
        if ncu.max_sm_tensor_ops is not None:
            max_rel_error = abs(model.max_sm_tensor_ops - ncu.max_sm_tensor_ops) / max(
                abs(model.max_sm_tensor_ops),
                abs(ncu.max_sm_tensor_ops),
                1.0,
            )
            max_match = max_rel_error <= args.max_relative_error

        result = {
            "device": args.device,
            "gpu_name": gpu.name,
            "m": case.m,
            "n": case.n,
            "k": case.k,
            "tb_m": case.tb_m,
            "tb_n": case.tb_n,
            "tb_k": case.tb_k,
            "split_k": case.split_k,
            "swizzle": case.swizzle,
            "dtype": case.dtype,
            "probe_task_count": reference_task_count,
            "model_task_count": model.task_count,
            "task_count_match": task_count_match,
            "model_total_tensor_ops": model.total_tensor_ops,
            "ncu_total_tensor_ops": ncu.total_tensor_ops,
            "total_tensor_ops_rel_error": total_rel_error,
            "total_tensor_ops_match": total_match,
            "model_max_sm_tensor_ops": model.max_sm_tensor_ops,
            "ncu_max_sm_tensor_ops": ncu.max_sm_tensor_ops,
            "max_sm_supported": ncu.max_sm_supported,
            "max_sm_source": ncu.max_sm_source,
            "max_sm_tensor_ops_rel_error": max_rel_error,
            "max_sm_tensor_ops_match": max_match,
            "all_metrics_match": task_count_match and total_match and max_match is True,
            "ncu_csv_path": str(ncu_csv_path),
        }
        results.append(result)

        print(
            f"[{index}/{len(cases)}] "
            f"M={case.m} N={case.n} K={case.k} "
            f"tb=({case.tb_m},{case.tb_n},{case.tb_k}) "
            f"split_k={case.split_k} swizzle={case.swizzle} "
            f"task_match={task_count_match} total_match={total_match} max_match={max_match}"
        )

        if args.stop_on_mismatch and not result["all_metrics_match"]:
            break

    summary = summarize_ncu_sweep_results(results)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "device",
            "gpu_name",
            "m",
            "n",
            "k",
            "tb_m",
            "tb_n",
            "tb_k",
            "split_k",
            "swizzle",
            "dtype",
            "probe_task_count",
            "model_task_count",
            "task_count_match",
            "model_total_tensor_ops",
            "ncu_total_tensor_ops",
            "total_tensor_ops_rel_error",
            "total_tensor_ops_match",
            "model_max_sm_tensor_ops",
            "ncu_max_sm_tensor_ops",
            "max_sm_supported",
            "max_sm_source",
            "max_sm_tensor_ops_rel_error",
            "max_sm_tensor_ops_match",
            "all_metrics_match",
            "ncu_csv_path",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow({name: result.get(name) for name in fieldnames})

    return 0 if summary["fully_matched_cases"] == summary["total_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
