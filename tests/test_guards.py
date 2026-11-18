import pandas as pd
import pytest

from fraud.config import load_config
from fraud.guards import assert_time_sorted, split_heldout


def _df():
    return pd.DataFrame({"t": [1, 2, 3, 4, 5], "y": [0, 1, 0, 0, 1]})


def test_unsorted_time_rejected():
    with pytest.raises(ValueError):
        assert_time_sorted(pd.DataFrame({"t": [2, 1]}), "t")


def test_heldout_hidden_without_flag(monkeypatch):
    monkeypatch.delenv("FINAL_EVAL", raising=False)
    dev, held = split_heldout(_df(), "t", heldout_start=4)
    assert held is None
    assert dev["t"].max() < 4


def test_heldout_returned_with_flag(monkeypatch):
    monkeypatch.setenv("FINAL_EVAL", "1")
    dev, held = split_heldout(_df(), "t", heldout_start=4)
    assert list(held["t"]) == [4, 5]
    assert len(dev) + len(held) == 5


def test_base_config_loads():
    cfg = load_config()
    assert cfg["label_delay_days"] == [0, 7, 30, 60]
