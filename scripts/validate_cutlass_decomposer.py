#!/usr/bin/env python3
"""Compare the repo's CUTLASS GEMM decomposer against a CUTLASS-backed probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perf_model.common.types import GemmProblem, KernelMeta
from perf_model.validation.cutlass_external import build_model_tasks, compare_task_lists, run_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--dump-reference-json", action="store_true")
    parser.add_argument("--max-diff", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problem = GemmProblem(M=args.m, N=args.n, K=args.k, split_k_slices=args.split_k)
    kernel_meta = KernelMeta(
        name="cutlass_validation",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(args.tb_m, args.tb_n, args.tb_k),
        warp_shape=(args.warp_m, args.warp_n, args.warp_k),
        instruction_shape=(args.inst_m, args.inst_n, args.inst_k),
        swizzle=args.swizzle,
        dtype=args.dtype,
    )

    reference = run_probe(
        PROJECT_ROOT,
        m=args.m,
        n=args.n,
        k=args.k,
        tb_m=args.tb_m,
        tb_n=args.tb_n,
        tb_k=args.tb_k,
        split_k=args.split_k,
        swizzle=args.swizzle,
        dtype=args.dtype,
        device=args.device,
    )
    model_tasks = build_model_tasks(problem, kernel_meta)
    comparison = compare_task_lists(reference["tasks"], model_tasks)

    print(
        f"device={reference['device']} name={reference['device_name']} "
        f"swizzle={reference['swizzle']} dtype={reference['dtype']}"
    )
    print(
        f"reference_tasks={len(reference['tasks'])} model_tasks={len(model_tasks)} "
        f"grid_shape={tuple(reference['grid_shape'])} "
        f"grid_tiled_shape={tuple(reference['grid_tiled_shape'])} "
        f"swizzle_log_tile={reference['swizzle_log_tile']} "
        f"gemm_k_size={reference['gemm_k_size']}"
    )
    print(f"match={comparison.is_match}")

    if not comparison.is_match:
        print("only_in_reference:")
        for task in comparison.only_in_reference[: args.max_diff]:
            print(json.dumps(task, sort_keys=True))
        print("only_in_model:")
        for task in comparison.only_in_model[: args.max_diff]:
            print(json.dumps(task, sort_keys=True))

    if args.dump_reference_json:
        print(json.dumps(reference, indent=2, sort_keys=True))

    return 0 if comparison.is_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
