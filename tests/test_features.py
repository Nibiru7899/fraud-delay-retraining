"""Hand-computed checks + no-future-leak property for point-in-time features."""
import numpy as np
import pandas as pd
import pytest

from fraud.features import add_uid, build_features, build_features_v2, past_key_features, past_means

D = 86400


def _frame():
    # key A: t=0 (10), t=100 (20), t=100 (30, same second), t=2D (40); key B: t=50 (5)
    return pd.DataFrame(
        {
            "key": ["A", "B", "A", "A", "A"],
            "t": [0, 50, 100, 100, 2 * D],
            "amt": [10.0, 5.0, 20.0, 30.0, 40.0],
        }
    )


def test_hand_computed_values():
    f = _frame()
    out = past_key_features(f["key"], f["t"], f["amt"], "k")
    # row 0: nothing before
    assert out.loc[0, "k_n_prev"] == 0
    assert np.isnan(out.loc[0, "k_amt_mean_prev"])
    assert np.isnan(out.loc[0, "k_secs_since_last"])
    # rows 2 and 3 share t=100: neither sees the other (strictly earlier only)
    for r in (2, 3):
        assert out.loc[r, "k_n_prev"] == 1
        assert out.loc[r, "k_amt_mean_prev"] == 10.0
        assert out.loc[r, "k_secs_since_last"] == 100
    assert out.loc[2, "k_amt_ratio"] == 2.0
    # row 4 at 2 days: prior A rows = 10, 20, 30
    assert out.loc[4, "k_n_prev"] == 3
    assert out.loc[4, "k_amt_mean_prev"] == pytest.approx(20.0)
    assert out.loc[4, "k_amt_std_prev"] == pytest.approx(10.0)  # sample std of 10,20,30
    assert out.loc[4, "k_amt_z"] == pytest.approx(2.0)
    assert out.loc[4, "k_cnt_1d"] == 0  # window [t-1d, t) excludes rows at t<=1d? t-1d = 1D
    assert out.loc[4, "k_cnt_7d"] == 3
    assert out.loc[4, "k_amt_sum_7d"] == 60.0
    assert out.loc[4, "k_secs_since_last"] == 2 * D - 100
    # key B only sees itself
    assert out.loc[1, "k_n_prev"] == 0


def test_window_boundary_inclusive_left():
    f = pd.DataFrame({"key": ["A", "A"], "t": [0, D], "amt": [1.0, 2.0]})
    out = past_key_features(f["key"], f["t"], f["amt"], "k")
    assert out.loc[1, "k_cnt_1d"] == 1  # row at exactly t-1d is inside [t-1d, t)


def test_missing_key_gives_nan():
    f = pd.DataFrame({"key": ["A", None, "A"], "t": [0, 1, 2], "amt": [1.0, 2.0, 3.0]})
    out = past_key_features(f["key"], f["t"], f["amt"], "k")
    assert out.loc[1].isna().all()
    assert out.loc[2, "k_n_prev"] == 1


def test_uid_construction():
    f = pd.DataFrame(
        {"TransactionDT": [10 * D, 12 * D, 12 * D], "card1": [1, 1, 1],
         "addr1": [5.0, 5.0, np.nan], "D1": [3.0, 5.0, 5.0]}
    )
    uid = add_uid(f)
    assert uid[0] == uid[1] == "1_5_7"  # day - D1 = 7 for both
    assert pd.isna(uid[2])


def _ieee_like(rng_seed=0, n=400):
    """Small hand-built IEEE-shaped frame for the leak property (not experiment data)."""
    r = np.random.default_rng(rng_seed)
    t = np.sort(r.integers(0, 40 * D, n))
    return pd.DataFrame(
        {
            "TransactionDT": t,
            "TransactionAmt": r.uniform(1, 500, n).astype("float32"),
            "card1": r.integers(1, 15, n),
            "addr1": r.choice([1.0, 2.0, np.nan], n),
            "D1": r.integers(0, 30, n).astype(float),
            "P_emaildomain": r.choice(["a.com", "b.com", None], n),
            "D4": r.choice([1.0, 5.0, np.nan], n),
            "C1": r.integers(0, 5, n).astype(float),
            "C13": np.where(r.uniform(size=n) < 0.2, np.nan, r.uniform(0, 9, n)),
            "card2": r.choice([100.0, 200.0, np.nan], n),
            "DeviceInfo": r.choice(["ios", "win", None], n),
        }
    )


@pytest.mark.parametrize("builder", [build_features, build_features_v2])
@pytest.mark.parametrize("cut_day", [5, 17, 33])
def test_no_future_leak(cut_day, builder):
    """Features for rows before a cut-off are identical whether or not later rows exist."""
    df = _ieee_like()
    cut = cut_day * D
    full = builder(df)
    trunc = builder(df[df["TransactionDT"] < cut])
    pd.testing.assert_frame_equal(full.loc[trunc.index], trunc, check_exact=True)


def test_past_means_hand():
    f = pd.DataFrame({"key": ["A", "A", "A", "B"], "t": [0, 10, 20, 5],
                      "v": [1.0, np.nan, 5.0, 9.0]})
    out = past_means(f["key"], f["t"], f[["v"]], "k")["k_v_mean_prev"]
    assert np.isnan(out[0]) and out[1] == 1.0 and out[2] == 1.0  # NaN skipped
    assert np.isnan(out[3])
