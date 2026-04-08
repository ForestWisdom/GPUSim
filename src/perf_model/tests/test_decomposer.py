from perf_model.common.types import GemmProblem, KernelMeta
from perf_model.kernel_desc.cutlass_gemm import CutlassGemmDecomposer


def test_cutlass_decomposer_generates_expected_number_of_tasks() -> None:
    problem = GemmProblem(M=256, N=128, K=64, split_k_slices=2)
    kernel = KernelMeta(
        name="cutlass_tensorop",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(128, 64, 32),
        warp_shape=(64, 32, 32),
        instruction_shape=(16, 8, 16),
    )

    tasks = CutlassGemmDecomposer().decompose(problem, kernel)

    assert len(tasks) == 8
    assert tasks[-1].k_eff == 32


def test_cutlass_decomposer_handles_edge_tiles() -> None:
    problem = GemmProblem(M=130, N=65, K=17)
    kernel = KernelMeta(
        name="cutlass_tensorop",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(128, 64, 16),
        warp_shape=(64, 32, 16),
        instruction_shape=(16, 8, 16),
    )

    tasks = CutlassGemmDecomposer().decompose(problem, kernel)

    assert tasks[-1].m_eff == 2
    assert tasks[-1].n_eff == 1
    assert tasks[-1].k_eff == 17
