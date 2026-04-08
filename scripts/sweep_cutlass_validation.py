#!/usr/bin/env python3
"""Run a batch CUTLASS decomposition correctness sweep."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perf_model.validation.cutlass_external import generate_cases, summarize_sweep_results, validate_case


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_tb_shapes(value: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for item in _parse_str_list(value):
        m_str, n_str, k_str = item.lower().split("x")
        shapes.append((int(m_str), int(n_str), int(k_str)))
    return shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--m-values", default="128,130,256,384,1024")
    parser.add_argument("--n-values", default="128,129,256,384,1024")
    parser.add_argument("--k-values", default="32,33,64,96,128,256")
    parser.add_argument("--tb-shapes", default="128x128x32,128x64x32")
    parser.add_argument("--split-k-values", default="1,2,4")
    parser.add_argument("--swizzles", default="Identity,Horizontal,Identity2,Identity4")
    parser.add_argument("--dtypes", default="f16")
    parser.add_argument("--output-json", help="Optional JSON output path")
    parser.add_argument("--output-csv", help="Optional CSV output path")
    parser.add_argument("--stop-on-mismatch", action="store_true")
    parser.add_argument("--max-diff", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = generate_cases(
        m_values=_parse_int_list(args.m_values),
        n_values=_parse_int_list(args.n_values),
        k_values=_parse_int_list(args.k_values),
        tb_shapes=_parse_tb_shapes(args.tb_shapes),
        split_k_values=_parse_int_list(args.split_k_values),
        swizzles=_parse_str_list(args.swizzles),
        dtypes=_parse_str_list(args.dtypes),
    )

    print(f"device={args.device} total_cases={len(cases)}")

    results: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        result = validate_case(PROJECT_ROOT, case=case, device=args.device, max_diff=args.max_diff)
        results.append(result)
        print(
            f"[{index}/{len(cases)}] "
            f"M={case.m} N={case.n} K={case.k} "
            f"tb=({case.tb_m},{case.tb_n},{case.tb_k}) "
            f"split_k={case.split_k} swizzle={case.swizzle} dtype={case.dtype} "
            f"match={result['match']}"
        )
        if args.stop_on_mismatch and not result["match"]:
            break

    summary = summarize_sweep_results(results)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "device",
            "device_name",
            "m",
            "n",
            "k",
            "tb_m",
            "tb_n",
            "tb_k",
            "split_k",
            "swizzle",
            "dtype",
            "match",
            "reference_tasks",
            "model_tasks",
            "swizzle_log_tile",
            "gemm_k_size",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow({name: result.get(name) for name in fieldnames})

    return 0 if summary["mismatched_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
