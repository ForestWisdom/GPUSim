"""Compare local decomposition against a CUTLASS-backed reference probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import product
import json
import subprocess

from perf_model.common.types import GemmProblem, KernelMeta, dataclass_to_dict
from perf_model.kernel_desc.cutlass_gemm import CutlassGemmDecomposer


TASK_SIGNATURE_FIELDS = (
    "tile_idx_m",
    "tile_idx_n",
    "tile_idx_k",
    "m0",
    "m1",
    "n0",
    "n1",
    "k0",
    "k1",
    "m_eff",
    "n_eff",
    "k_eff",
    "gemm_k_iterations",
)


@dataclass(slots=True)
class TaskListComparison:
    is_match: bool
    only_in_reference: list[dict[str, int]]
    only_in_model: list[dict[str, int]]


@dataclass(slots=True)
class ValidationCase:
    m: int
    n: int
    k: int
    tb_m: int
    tb_n: int
    tb_k: int
    split_k: int
    swizzle: str
    dtype: str


def _signature(task: dict[str, int]) -> tuple[int, ...]:
    return tuple(int(task[field]) for field in TASK_SIGNATURE_FIELDS)


def compare_task_lists(
    reference_tasks: list[dict[str, int]], model_tasks: list[dict[str, int]]
) -> TaskListComparison:
    reference_map = {_signature(task): task for task in reference_tasks}
    model_map = {_signature(task): task for task in model_tasks}

    only_in_reference_keys = sorted(set(reference_map) - set(model_map))
    only_in_model_keys = sorted(set(model_map) - set(reference_map))

    return TaskListComparison(
        is_match=not only_in_reference_keys and not only_in_model_keys,
        only_in_reference=[reference_map[key] for key in only_in_reference_keys],
        only_in_model=[model_map[key] for key in only_in_model_keys],
    )


def build_model_tasks(problem: GemmProblem, kernel_meta: KernelMeta) -> list[dict[str, int]]:
    tasks = CutlassGemmDecomposer().decompose(problem, kernel_meta)
    return [
        {
            key: int(value) if isinstance(value, bool | int) else value
            for key, value in dataclass_to_dict(task).items()
        }
        for task in tasks
    ]


def build_probe_binary(repo_root: Path) -> Path:
    probe_source = repo_root / "tools" / "cutlass_grid_probe.cu"
    build_dir = repo_root / ".cache"
    build_dir.mkdir(exist_ok=True)
    binary_path = build_dir / "cutlass_grid_probe"

    if binary_path.exists() and binary_path.stat().st_mtime >= probe_source.stat().st_mtime:
        return binary_path

    command = [
        "nvcc",
        "-std=c++17",
        "-O2",
        "--expt-relaxed-constexpr",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_89,code=compute_89",
        f"-I{repo_root / 'thirdparty' / 'cutlass' / 'include'}",
        str(probe_source),
        "-o",
        str(binary_path),
    ]
    subprocess.run(command, check=True, cwd=repo_root)
    return binary_path


def run_probe(
    repo_root: Path,
    *,
    m: int,
    n: int,
    k: int,
    tb_m: int,
    tb_n: int,
    tb_k: int,
    split_k: int,
    swizzle: str,
    dtype: str,
    device: int,
) -> dict[str, object]:
    binary_path = build_probe_binary(repo_root)
    command = [
        str(binary_path),
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--tb-m",
        str(tb_m),
        "--tb-n",
        str(tb_n),
        "--tb-k",
        str(tb_k),
        "--split-k",
        str(split_k),
        "--swizzle",
        swizzle,
        "--dtype",
        dtype,
        "--device",
        str(device),
    ]
    result = subprocess.run(command, check=True, cwd=repo_root, capture_output=True, text=True)
    return json.loads(result.stdout)


def validate_case(
    repo_root: Path,
    *,
    case: ValidationCase,
    device: int,
    warp_m: int = 64,
    warp_n: int = 64,
    warp_k: int = 32,
    inst_m: int = 16,
    inst_n: int = 8,
    inst_k: int = 16,
    max_diff: int = 10,
) -> dict[str, object]:
    problem = GemmProblem(M=case.m, N=case.n, K=case.k, split_k_slices=case.split_k)
    kernel_meta = KernelMeta(
        name="cutlass_validation",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(case.tb_m, case.tb_n, case.tb_k),
        warp_shape=(warp_m, warp_n, warp_k),
        instruction_shape=(inst_m, inst_n, inst_k),
        swizzle=case.swizzle,
        dtype=case.dtype,
    )
    reference = run_probe(
        repo_root,
        m=case.m,
        n=case.n,
        k=case.k,
        tb_m=case.tb_m,
        tb_n=case.tb_n,
        tb_k=case.tb_k,
        split_k=case.split_k,
        swizzle=case.swizzle,
        dtype=case.dtype,
        device=device,
    )
    model_tasks = build_model_tasks(problem, kernel_meta)
    comparison = compare_task_lists(reference["tasks"], model_tasks)
    return {
        "device": device,
        "device_name": reference["device_name"],
        "m": case.m,
        "n": case.n,
        "k": case.k,
        "tb_m": case.tb_m,
        "tb_n": case.tb_n,
        "tb_k": case.tb_k,
        "split_k": case.split_k,
        "swizzle": case.swizzle,
        "dtype": case.dtype,
        "match": comparison.is_match,
        "reference_tasks": len(reference["tasks"]),
        "model_tasks": len(model_tasks),
        "grid_shape": reference["grid_shape"],
        "grid_tiled_shape": reference["grid_tiled_shape"],
        "swizzle_log_tile": reference["swizzle_log_tile"],
        "gemm_k_size": reference["gemm_k_size"],
        "only_in_reference": comparison.only_in_reference[:max_diff],
        "only_in_model": comparison.only_in_model[:max_diff],
    }


def generate_cases(
    *,
    m_values: list[int],
    n_values: list[int],
    k_values: list[int],
    tb_shapes: list[tuple[int, int, int]],
    split_k_values: list[int],
    swizzles: list[str],
    dtypes: list[str],
) -> list[ValidationCase]:
    cases: list[ValidationCase] = []
    for m, n, k, (tb_m, tb_n, tb_k), split_k, swizzle, dtype in product(
        m_values, n_values, k_values, tb_shapes, split_k_values, swizzles, dtypes
    ):
        cases.append(
            ValidationCase(
                m=m,
                n=n,
                k=k,
                tb_m=tb_m,
                tb_n=tb_n,
                tb_k=tb_k,
                split_k=split_k,
                swizzle=swizzle,
                dtype=dtype,
            )
        )
    return cases


def summarize_sweep_results(results: list[dict[str, object]]) -> dict[str, object]:
    total_cases = len(results)
    matched_cases = sum(1 for item in results if item["match"])
    mismatched_cases = total_cases - matched_cases
    groups: dict[tuple[object, ...], int] = {}

    for item in results:
        if item["match"]:
            continue
        key = (
            item["swizzle"],
            item["dtype"],
            item.get("tb_m"),
            item.get("tb_n"),
            item.get("tb_k"),
            item.get("split_k"),
        )
        groups[key] = groups.get(key, 0) + 1

    mismatch_groups = [
        {
            "swizzle": key[0],
            "dtype": key[1],
            "tb_m": key[2],
            "tb_n": key[3],
            "tb_k": key[4],
            "split_k": key[5],
            "count": count,
        }
        for key, count in sorted(groups.items(), key=lambda item: (-item[1], item[0]))
    ]

    return {
        "total_cases": total_cases,
        "matched_cases": matched_cases,
        "mismatched_cases": mismatched_cases,
        "match_rate": (matched_cases / total_cases) if total_cases else 0.0,
        "mismatch_groups": mismatch_groups,
    }
