"""Compare analytical tensor-op summaries against Nsight Compute exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from perf_model.common.types import GemmProblem, GpuSpec, KernelMeta
from perf_model.common.utils import load_yaml
from perf_model.pipelines.feature_pipeline import build_default_feature_pipeline
from perf_model.profiling.ncu_parser import extract_metric_summary, parse_ncu_report
from perf_model.validation.cutlass_external import run_probe


def _safe_relative_error(lhs: float, rhs: float) -> float:
    scale = max(abs(lhs), abs(rhs), 1.0)
    return abs(lhs - rhs) / scale


@dataclass(slots=True)
class ModelTensorSummary:
    task_count: int
    total_tensor_ops: float
    max_sm_tensor_ops: float


@dataclass(slots=True)
class NcuTensorSummary:
    metric_name: str
    total_tensor_ops: float
    max_sm_tensor_ops: float | None
    max_sm_supported: bool
    max_sm_source: str
    instance_count: int
    kernel_names: list[str]


def load_gpu_spec(path: str | Path) -> GpuSpec:
    data = load_yaml(path)
    return GpuSpec(
        name=str(data["name"]),
        num_sms=int(data["num_sms"]),
        tensor_throughput_per_sm=float(data["tensor_throughput_per_sm"]),
        simt_throughput_per_sm=float(data["simt_throughput_per_sm"]),
        dram_bw_gbps=float(data["dram_bw_gbps"]),
        l2_bw_gbps=float(data["l2_bw_gbps"]),
        smem_bw_gbps_per_sm=float(data["smem_bw_gbps_per_sm"]),
        clock_mhz=float(data["clock_mhz"]),
    )


def build_model_tensor_summary(
    problem: GemmProblem,
    gpu: GpuSpec,
    kernel_meta: KernelMeta,
) -> ModelTensorSummary:
    pipeline = build_default_feature_pipeline(kernel_meta)
    _, debug = pipeline.run(problem, gpu, kernel_meta)
    sm_features = debug["sm_features"]

    return ModelTensorSummary(
        task_count=len(debug["tasks"]),
        total_tensor_ops=float(sum(item.total_tensor_ops for item in sm_features)),
        max_sm_tensor_ops=float(max((item.total_tensor_ops for item in sm_features), default=0.0)),
    )


def build_ncu_tensor_summary(
    ncu_csv_path: str | Path,
    *,
    metric_name: str,
    metric_scale: float = 1.0,
    max_metric_name: str | None = None,
    max_metric_scale: float | None = None,
    model_task_count: int | None = None,
) -> NcuTensorSummary:
    report = parse_ncu_report(ncu_csv_path)
    summary = extract_metric_summary(report, metric_name)
    total_tensor_ops = summary.total * metric_scale
    if max_metric_name is not None:
        max_summary = extract_metric_summary(report, max_metric_name)
        max_sm_tensor_ops = max_summary.total * (
            max_metric_scale if max_metric_scale is not None else metric_scale
        )
        max_sm_supported = True
        max_sm_source = f"metric:{max_metric_name}"
    elif summary.instance_count > 1:
        max_sm_tensor_ops: float | None = summary.max_instance * metric_scale
        max_sm_supported = True
        max_sm_source = "instances"
    elif model_task_count == 1:
        max_sm_tensor_ops = total_tensor_ops
        max_sm_supported = True
        max_sm_source = "single_task_total"
    else:
        max_sm_tensor_ops = None
        max_sm_supported = False
        max_sm_source = "aggregate_only"
    return NcuTensorSummary(
        metric_name=summary.metric_name,
        total_tensor_ops=total_tensor_ops,
        max_sm_tensor_ops=max_sm_tensor_ops,
        max_sm_supported=max_sm_supported,
        max_sm_source=max_sm_source,
        instance_count=summary.instance_count,
        kernel_names=report.kernel_names,
    )


def probe_task_count(
    repo_root: Path,
    *,
    problem: GemmProblem,
    kernel_meta: KernelMeta,
    device: int,
) -> int:
    reference = run_probe(
        repo_root,
        m=problem.M,
        n=problem.N,
        k=problem.K,
        tb_m=kernel_meta.threadblock_shape[0],
        tb_n=kernel_meta.threadblock_shape[1],
        tb_k=kernel_meta.threadblock_shape[2],
        split_k=problem.split_k_slices,
        swizzle=kernel_meta.swizzle,
        dtype=kernel_meta.dtype,
        device=device,
    )
    return len(reference["tasks"])


def build_comparison_payload(
    *,
    model: ModelTensorSummary,
    ncu: NcuTensorSummary | None = None,
    reference_task_count: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"model": asdict(model)}
    if reference_task_count is not None:
        payload["probe"] = {
            "task_count": reference_task_count,
            "task_count_match": reference_task_count == model.task_count,
        }
    if ncu is not None:
        payload["ncu"] = asdict(ncu)
        payload["relative_error"] = {
            "total_tensor_ops": _safe_relative_error(model.total_tensor_ops, ncu.total_tensor_ops),
            "max_sm_tensor_ops": (
                _safe_relative_error(model.max_sm_tensor_ops, ncu.max_sm_tensor_ops)
                if ncu.max_sm_tensor_ops is not None
                else None
            ),
        }
    return payload
