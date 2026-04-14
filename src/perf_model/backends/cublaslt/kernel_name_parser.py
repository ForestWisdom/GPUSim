"""Parse useful metadata from cuBLAS kernel names."""

from __future__ import annotations

import re

_TB_PATTERN = re.compile(r"_(\d+)x(\d+)_")
_STAGES_PATTERN = re.compile(r"_stages_(\d+)x\d+_")
_LAYOUT_PATTERN = re.compile(r"_([nt]{2})$")


def parse_cublas_kernel_name(kernel_name: str) -> dict[str, int | str | None]:
    tb_match = _TB_PATTERN.search(kernel_name)
    stages_match = _STAGES_PATTERN.search(kernel_name)
    layout_match = _LAYOUT_PATTERN.search(kernel_name)

    family_match = re.match(r"([A-Za-z0-9]+(?:_[A-Za-z0-9]+gemm)?)", kernel_name)
    kernel_family = family_match.group(1) if family_match else kernel_name

    return {
        "kernel_family": kernel_family,
        "threadblock_m": int(tb_match.group(1)) if tb_match else None,
        "threadblock_n": int(tb_match.group(2)) if tb_match else None,
        "stages": int(stages_match.group(1)) if stages_match else None,
        "layout_tag": layout_match.group(1) if layout_match else None,
        "instruction_family": "tensor_core",
    }
