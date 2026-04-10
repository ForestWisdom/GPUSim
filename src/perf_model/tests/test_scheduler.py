from perf_model.common.types import GemmProblem, GpuSpec, KernelMeta
from perf_model.kernel_desc.cutlass_gemm import CutlassGemmDecomposer
from perf_model.scheduler.round_robin import RoundRobinScheduler


def test_round_robin_assigns_every_task_once() -> None:
    problem = GemmProblem(M=256, N=128, K=64)
    kernel = KernelMeta(
        name="cutlass_tensorop",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(128, 64, 32),
        warp_shape=(64, 32, 32),
        instruction_shape=(16, 8, 16),
    )
    gpu = GpuSpec(
        name="toy",
        num_sms=3,
        tensor_throughput_per_sm=1.0,
        simt_throughput_per_sm=1.0,
        dram_bw_bytes_per_cycle=1.0,
        l2_bw_bytes_per_cycle=1.0,
        smem_bw_bytes_per_cycle_per_sm=1.0,
        clock_mhz=1000,
    )

    tasks = CutlassGemmDecomposer().decompose(problem, kernel)
    assignments = RoundRobinScheduler().assign(tasks, gpu)

    assigned_task_ids = sorted(task.task_idx for bucket in assignments.values() for task in bucket)
    assert assigned_task_ids == list(range(len(tasks)))
