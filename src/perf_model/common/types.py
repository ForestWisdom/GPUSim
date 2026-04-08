"""Core project data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class GemmProblem:
    M: int
    N: int
    K: int
    split_k_slices: int = 1


@dataclass(slots=True)
class KernelMeta:
    name: str
    backend: str
    pipeline: str
    threadblock_shape: tuple[int, int, int]
    warp_shape: tuple[int, int, int]
    instruction_shape: tuple[int, int, int]
    swizzle: str = "Identity"
    split_k_default: int = 1
    dtype: str = "f16"
    extra: dict[str, int | float | str] = field(default_factory=dict)


@dataclass(slots=True)
class GpuSpec:
    name: str
    num_sms: int
    tensor_throughput_per_sm: float
    simt_throughput_per_sm: float
    dram_bw_gbps: float
    l2_bw_gbps: float
    smem_bw_gbps_per_sm: float
    clock_mhz: float


@dataclass(slots=True)
class GemmTask:
    tile_m: int
    tile_n: int
    tile_k: int
    m0: int
    m1: int
    n0: int
    n1: int
    k0: int
    k1: int
    m_eff: int
    n_eff: int
    k_eff: int
    gemm_k_iterations: int
    task_idx: int
    tile_idx_m: int
    tile_idx_n: int
    tile_idx_k: int


@dataclass(slots=True)
class TaskFeatures:
    task_idx: int
    sm_id: int
    math_ops: float
    math_cycles: float
    bytes_a: float
    bytes_b: float
    bytes_c: float
    memory_cycles: float


@dataclass(slots=True)
class SmFeatures:
    sm_id: int
    task_count: int
    total_math_ops: float
    total_math_cycles: float
    total_memory_cycles: float
    estimated_busy_cycles: float


def dataclass_to_dict(value: object) -> dict[str, object]:
    return asdict(value)
