"""Standardize latency CSV reads."""

from __future__ import annotations

import pandas as pd


def read_latency_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
