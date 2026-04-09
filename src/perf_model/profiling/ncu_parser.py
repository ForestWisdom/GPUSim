"""Helpers for parsing Nsight Compute CSV exports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


def _to_float(raw: str) -> float:
    value = raw.strip().strip('"')
    if not value:
        return 0.0
    return float(value.replace(",", ""))


def _parse_instances(raw: str) -> list[float]:
    value = raw.strip().strip('"')
    if not value:
        return []
    if "(" in value and value.endswith(")"):
        _, instance_blob = value.split("(", 1)
        values: list[float] = []
        for item in instance_blob[:-1].split(";"):
            if not item.strip():
                continue
            try:
                values.append(_to_float(item))
            except ValueError:
                pass
        return values
    values = []
    for item in value.split(";"):
        if not item.strip():
            continue
        try:
            values.append(_to_float(item))
        except ValueError:
            pass
    return values


@dataclass(slots=True)
class NcuMetricRow:
    kernel_name: str
    metric_name: str
    metric_unit: str
    metric_value: float
    metric_instances: list[float]


@dataclass(slots=True)
class NcuReport:
    rows: list[NcuMetricRow]
    kernel_names: list[str]


@dataclass(slots=True)
class NcuMetricSummary:
    metric_name: str
    total: float
    max_instance: float
    instance_count: int
    instances: list[float]


def _parse_long_form(rows: list[list[str]]) -> NcuReport | None:
    header = rows[0]
    if "Metric Name" not in header:
        return None

    reader = csv.DictReader([",".join(header)] + [",".join(row) for row in rows[1:]])
    parsed_rows: list[NcuMetricRow] = []
    kernel_names: list[str] = []

    for raw_row in reader:
        metric_name = (raw_row.get("Metric Name") or "").strip().strip('"')
        if not metric_name:
            continue

        kernel_name = (raw_row.get("Kernel Name") or "").strip().strip('"')
        metric_unit = (raw_row.get("Metric Unit") or "").strip().strip('"')
        metric_value = _to_float(raw_row.get("Metric Value") or "0")
        metric_instances = _parse_instances(raw_row.get("Metric Instances") or "")

        parsed_rows.append(
            NcuMetricRow(
                kernel_name=kernel_name,
                metric_name=metric_name,
                metric_unit=metric_unit,
                metric_value=metric_value,
                metric_instances=metric_instances,
            )
        )
        if kernel_name and kernel_name not in kernel_names:
            kernel_names.append(kernel_name)

    return NcuReport(rows=parsed_rows, kernel_names=kernel_names)


def _parse_wide_form(rows: list[list[str]]) -> NcuReport:
    header = rows[0]
    unit_row = rows[1] if len(rows) > 1 else []
    data_rows = rows[2:] if len(rows) > 2 else []

    kernel_idx = header.index("Kernel Name") if "Kernel Name" in header else -1
    parsed_rows: list[NcuMetricRow] = []
    kernel_names: list[str] = []

    for raw_row in data_rows:
        if not raw_row:
            continue
        kernel_name = raw_row[kernel_idx].strip().strip('"') if kernel_idx >= 0 else ""
        if kernel_name and kernel_name not in kernel_names:
            kernel_names.append(kernel_name)

        for idx, metric_name in enumerate(header):
            if "__" not in metric_name:
                continue
            raw_value = raw_row[idx] if idx < len(raw_row) else ""
            try:
                metric_value = _to_float(raw_value.split("(", 1)[0] if raw_value else "0")
            except ValueError:
                continue
            metric_instances = _parse_instances(raw_value)
            metric_unit = unit_row[idx].strip().strip('"') if idx < len(unit_row) else ""
            parsed_rows.append(
                NcuMetricRow(
                    kernel_name=kernel_name,
                    metric_name=metric_name,
                    metric_unit=metric_unit,
                    metric_value=metric_value,
                    metric_instances=metric_instances,
                )
            )

    return NcuReport(rows=parsed_rows, kernel_names=kernel_names)


def parse_ncu_report(path: str | Path) -> NcuReport:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    csv_start = next((idx for idx, line in enumerate(lines) if line.startswith('"ID"')), None)
    if csv_start is None:
        raise ValueError(f"could not find CSV header in {path}")

    rows = list(csv.reader(lines[csv_start:]))
    long_form = _parse_long_form(rows)
    if long_form is not None:
        return long_form
    return _parse_wide_form(rows)


def extract_metric_summary(report: NcuReport, metric_name: str) -> NcuMetricSummary:
    matching_rows = [row for row in report.rows if row.metric_name == metric_name]
    if not matching_rows:
        raise KeyError(f"metric not found in report: {metric_name}")

    instances: list[float] = []
    total = 0.0
    for row in matching_rows:
        total += row.metric_value
        instances.extend(row.metric_instances)

    fallback_instances = instances or [row.metric_value for row in matching_rows]
    return NcuMetricSummary(
        metric_name=metric_name,
        total=total,
        max_instance=max(fallback_instances) if fallback_instances else 0.0,
        instance_count=len(fallback_instances),
        instances=fallback_instances,
    )
