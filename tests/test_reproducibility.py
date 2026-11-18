"""Same seed + same data -> identical model output and metrics (plan: reproducibility test).

Uses a small random fixture (unit-test input only, never experiment data).
"""
import numpy as np
import pandas as pd

from fraud.metrics import score_all
from fraud.models import predict_lgb, train_lgb


def _data(n=3000, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": r.normal(size=n), "b": r.normal(size=n),
        "c": r.choice(["x", "y", "z"], n),
    })
    X.loc[r.uniform(size=n) < 0.1, "a"] = np.nan
    logit = 1.5 * X["b"] + (X["c"] == "x") * 1.0 - 3
    y = (r.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int).to_numpy()
    return X, y


def _fit_predict(seed):
    X, y = _data()
    tr, va = slice(0, 2000), slice(2000, 3000)
    booster, enc = train_lgb(X.iloc[tr], y[tr], X.iloc[va], y[va], seed,
                             max_rounds=200, patience=20)
    return predict_lgb(booster, enc, X.iloc[va]), y[va]


def test_same_seed_identical():
    p1, y = _fit_predict(7)
    p2, _ = _fit_predict(7)
    np.testing.assert_array_equal(p1, p2)
    assert score_all(y, p1) == score_all(y, p2)
