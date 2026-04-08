"""Deterministic dataset splitting."""

from __future__ import annotations

import pandas as pd

from perf_model.common.constants import DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO


def split_frame(
    frame: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy(), frame.copy()
    train_end = int(len(frame) * train_ratio)
    val_end = train_end + int(len(frame) * val_ratio)
    return (
        frame.iloc[:train_end].reset_index(drop=True),
        frame.iloc[train_end:val_end].reset_index(drop=True),
        frame.iloc[val_end:].reset_index(drop=True),
    )
