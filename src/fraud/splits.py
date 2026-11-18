"""Time-window selection. Any window reaching into the held-out tail needs FINAL_EVAL=1 (rule 2)."""
from __future__ import annotations

import itertools

import pandas as pd

from fraud.guards import assert_time_sorted, final_eval_enabled


def time_window(
    df: pd.DataFrame, start: float, end: float, heldout_start: float, time_col: str = "t_days"
) -> pd.DataFrame:
    """Rows with start <= t < end. Refuses to return held-out rows unless FINAL_EVAL=1."""
    if start >= end:
        raise ValueError(f"empty window [{start}, {end})")
    if end > heldout_start and not final_eval_enabled():
        raise PermissionError(
            f"window [{start}, {end}) reaches held-out tail (>= {heldout_start}); set FINAL_EVAL=1"
        )
    assert_time_sorted(df, time_col)
    t = df[time_col]
    return df.loc[(t >= start) & (t < end)]


def check_disjoint_ordered(windows: list[tuple[float, float]]) -> None:
    """Train/valid/test windows must be in time order and non-overlapping (rule 3)."""
    for (a0, a1), (b0, b1) in itertools.pairwise(windows):
        if not (a0 < a1 <= b0 < b1):
            raise ValueError(f"windows overlap or are out of order: {(a0, a1)} vs {(b0, b1)}")
