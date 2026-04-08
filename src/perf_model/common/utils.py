"""Small utilities shared by modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def ceil_div(x: int, y: int) -> int:
    if y <= 0:
        raise ValueError("divisor must be positive")
    return (x + y - 1) // y


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data
