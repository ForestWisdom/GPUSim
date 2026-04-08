"""Dataset loading helpers."""

from __future__ import annotations

import pandas as pd


def load_csv_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
