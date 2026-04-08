#!/usr/bin/env python3
"""Inspect CUTLASS GEMM decomposition for a single problem."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perf_model.backends.cutlass.partition import compute_cutlass_k_partition
from perf_model.common.types import GemmProblem, KernelMeta
from perf_model.kernel_desc.cutlass_gemm import CutlassGemmDecomposer


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
    parser.add_argument("--hide-tasks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem = GemmProblem(M=args.m, N=args.n, K=args.k, split_k_slices=args.split_k)
    kernel_meta = KernelMeta(
        name="debug_cutlass_gemm",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(args.tb_m, args.tb_n, args.tb_k),
        warp_shape=(args.warp_m, args.warp_n, args.warp_k),
        instruction_shape=(args.inst_m, args.inst_n, args.inst_k),
        swizzle=args.swizzle,
        dtype=args.dtype,
    )

    partition = compute_cutlass_k_partition(problem, kernel_meta)
    tasks = CutlassGemmDecomposer().decompose(problem, kernel_meta)
    logical_tiles: dict[tuple[int, int], list] = defaultdict(list)
    empty_tasks = 0
    duplicate_logical_tasks = 0
    seen_triplets: set[tuple[int, int, int]] = set()

    for task in tasks:
        if task.m_eff <= 0 or task.n_eff <= 0 or task.k_eff <= 0:
            empty_tasks += 1
        logical_tiles[(task.tile_idx_m, task.tile_idx_n)].append(task)
        logical_triplet = (task.tile_idx_m, task.tile_idx_n, task.tile_idx_k)
        if logical_triplet in seen_triplets:
            duplicate_logical_tasks += 1
        seen_triplets.add(logical_triplet)

    repeated_output_tiles = sum(1 for tile_tasks in logical_tiles.values() if len(tile_tasks) > 1)

    print(
        f"problem=M{problem.M} N{problem.N} K{problem.K} "
        f"tb=({args.tb_m},{args.tb_n},{args.tb_k}) "
        f"split_k={problem.split_k_slices} swizzle={args.swizzle}"
    )
    print(
        f"k_align={partition.k_align} gemm_k_size={partition.gemm_k_size} "
        f"effective_split_k={partition.effective_split_k}"
    )
    print(f"task_count={len(tasks)}")
    print(f"logical_output_tiles={len(logical_tiles)}")
    print(f"empty_tasks={empty_tasks}")
    print(f"repeated_output_tiles={repeated_output_tiles}")
    print(f"duplicate_logical_tasks={duplicate_logical_tasks}")

    if args.hide_tasks:
        return

    print(
        "task_idx tile_idx(m,n,k) output_range(m,n,k) "
        "effective(m,n,k) gemm_k_iterations edge(m,n,k)"
    )
    for task in tasks:
        print(
            f"{task.task_idx:03d} "
            f"({task.tile_idx_m},{task.tile_idx_n},{task.tile_idx_k}) "
            f"m[{task.m0}:{task.m1}) n[{task.n0}:{task.n1}) k[{task.k0}:{task.k1}) "
            f"eff=({task.m_eff},{task.n_eff},{task.k_eff}) "
            f"iters={task.gemm_k_iterations} "
            f"edge=({int(task.is_edge_m)},{int(task.is_edge_n)},{int(task.is_edge_k)})"
        )


if __name__ == "__main__":
    main()
