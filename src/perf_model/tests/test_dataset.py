from perf_model.common.types import GpuSpec, KernelMeta
from perf_model.dataset.builder import DatasetBuilder
from perf_model.pipelines.feature_pipeline import build_default_feature_pipeline


def test_dataset_builder_aligns_features_and_labels() -> None:
    gpu = GpuSpec(
        name="toy",
        num_sms=4,
        tensor_throughput_per_sm=64.0,
        simt_throughput_per_sm=32.0,
        dram_bw_gbps=1.0,
        l2_bw_gbps=1.0,
        smem_bw_gbps_per_sm=128.0,
        clock_mhz=1000.0,
    )
    kernel = KernelMeta(
        name="cutlass_tensorop",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(128, 128, 32),
        warp_shape=(64, 64, 32),
        instruction_shape=(16, 8, 16),
    )
    records = [
        {"M": 128, "N": 128, "K": 64, "latency_us": 10.0},
        {"M": 256, "N": 128, "K": 64, "latency_us": 18.0},
    ]

    builder = DatasetBuilder(build_default_feature_pipeline(kernel))
    frame = builder.build_frame(records, gpu, kernel)

    assert len(frame) == 2
    assert "latency_us" in frame.columns
    assert len([column for column in frame.columns if column.startswith("f_")]) > 0


def test_dataset_builder_uses_row_level_split_k_and_swizzle_overrides() -> None:
    gpu = GpuSpec(
        name="toy",
        num_sms=4,
        tensor_throughput_per_sm=64.0,
        simt_throughput_per_sm=32.0,
        dram_bw_gbps=1.0,
        l2_bw_gbps=1.0,
        smem_bw_gbps_per_sm=128.0,
        clock_mhz=1000.0,
    )
    kernel = KernelMeta(
        name="cutlass_tensorop",
        backend="cutlass",
        pipeline="tensor_core",
        threadblock_shape=(128, 128, 32),
        warp_shape=(64, 64, 32),
        instruction_shape=(16, 8, 16),
    )
    records = [
        {
            "M": 128,
            "N": 128,
            "K": 64,
            "split_k_slices": 2,
            "swizzle": "Identity4",
            "latency_us": 10.0,
        },
    ]

    builder = DatasetBuilder(build_default_feature_pipeline(kernel))
    frame = builder.build_frame(records, gpu, kernel)

    assert frame["split_k_slices"].tolist() == [2]
    assert frame["swizzle"].tolist() == ["Identity4"]
