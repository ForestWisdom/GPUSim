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
    tensor_throughput_per_sm: float   # ops / cycle / SM
    simt_throughput_per_sm: float     # ops / cycle / SM
    dram_bw_gbps: float               # interpreted as bytes / cycle in current model
    l2_bw_gbps: float                 # interpreted as bytes / cycle in current model
    smem_bw_gbps_per_sm: float        # interpreted as bytes / cycle / SM
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

    is_edge_m: bool = False
    is_edge_n: bool = False
    is_edge_k: bool = False


@dataclass(slots=True)
class TaskFeatures:
    task_idx: int
    sm_id: int

    tensor_ops: float
    tensor_cycles: float

    bytes_a: float
    bytes_b: float
    bytes_c: float
    bytes_global: float

    global_cycles: float
    l2_cycles: float
    smem_cycles: float


@dataclass(slots=True)
class SmFeatures:
    sm_id: int
    task_count: int

    total_tensor_ops: float
    total_tensor_cycles: float

    total_bytes_global: float

    total_global_cycles: float
    total_l2_cycles: float
    total_smem_cycles: float

    estimated_busy_cycles: float


def dataclass_to_dict(value: object) -> dict[str, object]:
    return asdict(value)