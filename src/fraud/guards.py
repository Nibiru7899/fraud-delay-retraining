"""Guards that enforce the accuracy rules in context.md (rules 2 and 3)."""
from __future__ import annotations

import os

import pandas as pd

FINAL_EVAL_FLAG = "FINAL_EVAL"


def final_eval_enabled() -> bool:
    return os.environ.get(FINAL_EVAL_FLAG) == "1"


def assert_time_sorted(df: pd.DataFrame, time_col: str) -> None:
    """Fail if rows are not in non-decreasing time order (rule 3: time order only)."""
    if not df[time_col].is_monotonic_increasing:
        raise ValueError(f"{time_col} is not sorted ascending; sort by time before splitting")


def split_heldout(
    df: pd.DataFrame, time_col: str, heldout_start: float
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Split into (development, heldout) at a time cut-off.

    Rows with time >= heldout_start are the held-out tail. It is returned only when
    FINAL_EVAL=1 is set; otherwise it is None, so tuning code can never see it (rule 2).
    """
    assert_time_sorted(df, time_col)
    is_heldout = df[time_col] >= heldout_start
    dev = df.loc[~is_heldout]
    heldout = df.loc[is_heldout] if final_eval_enabled() else None
    return dev, heldout
