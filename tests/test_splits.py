import pandas as pd
import pytest

from fraud.config import load_config
from fraud.splits import check_disjoint_ordered, time_window


def _df():
    return pd.DataFrame({"t_days": [0.0, 1.0, 2.0, 3.0, 4.0]})


def test_window_half_open():
    out = time_window(_df(), 1, 3, heldout_start=4)
    assert list(out["t_days"]) == [1.0, 2.0]


def test_window_into_heldout_blocked(monkeypatch):
    monkeypatch.delenv("FINAL_EVAL", raising=False)
    with pytest.raises(PermissionError):
        time_window(_df(), 2, 5, heldout_start=4)


def test_window_into_heldout_allowed_with_flag(monkeypatch):
    monkeypatch.setenv("FINAL_EVAL", "1")
    assert len(time_window(_df(), 2, 5, heldout_start=4)) == 3


def test_overlap_rejected():
    with pytest.raises(ValueError):
        check_disjoint_ordered([(0, 2), (1, 3)])


def test_configured_ieee_baseline_windows_are_valid():
    ieee = load_config()["datasets"]["ieee"]
    b = ieee["baseline"]
    wins = [tuple(b["train"]), tuple(b["valid"]), tuple(b["test"])]
    check_disjoint_ordered(wins)
    assert wins[-1][1] <= ieee["heldout_start"]
