"""Business metric: money caught minus review cost, and threshold selection on validation.

net_value(threshold) = sum(amount of flagged frauds) - review_cost * (#flagged)
Missed fraud is not subtracted separately (it is the gap to the maximum catchable amount).
"""
from __future__ import annotations

import numpy as np


def net_value(y, p, amt, threshold: float, review_cost: float) -> float:
    flag = p >= threshold
    return float(amt[flag & (y == 1)].sum() - review_cost * flag.sum())


def best_threshold(y, p, amt, review_cost: float) -> tuple[float, float]:
    """Exact search over all distinct score cut-offs (flag rows with p >= cut). O(n log n)."""
    y, p, amt = np.asarray(y), np.asarray(p), np.asarray(amt, dtype=np.float64)
    order = np.argsort(-p, kind="stable")
    ps, gain = p[order], np.where(y[order] == 1, amt[order], 0.0) - review_cost
    cum = np.cumsum(gain)
    # only cut where the next score differs (all ties are flagged together)
    last_of_tie = np.r_[ps[1:] != ps[:-1], True]
    cand = np.flatnonzero(last_of_tie)
    i = cand[np.argmax(cum[cand])]
    if cum[i] <= 0:
        return float("inf"), 0.0  # flagging nothing is best
    return float(ps[i]), float(cum[i])
