"""Point-in-time features: every value for a row at time t uses ONLY rows of the same key with
time strictly < t (rows at the same second are excluded too). No labels are used.

Implementation: rows are sorted by (key code, time) and encoded as one int64
`composite = code * BIG + time`. `searchsorted` on that array then gives, per row, the slice of
strictly-earlier rows of the same key. Exact integer arithmetic, no rolling-window edge cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BIG = np.int64(10**9)  # > max TransactionDT (~1.6e7 s), so keys never overlap
WINDOWS_S = {"1d": 86400, "7d": 7 * 86400, "30d": 30 * 86400}


def add_uid(df: pd.DataFrame) -> pd.Series:
    """Client proxy: card1 + addr1 + account-start day (day - D1). NaN if a part is missing.

    D1 = days since client began; (transaction day - D1) is constant per client.
    """
    day = np.floor(df["TransactionDT"] / 86400)
    start = (day - df["D1"]).round()
    ok = df["addr1"].notna() & df["D1"].notna()
    uid = (
        df["card1"].astype("int64").astype(str)
        + "_" + df["addr1"].fillna(-1).astype("int64").astype(str)
        + "_" + start.fillna(-1).astype("int64").astype(str)
    )
    return uid.where(ok)


def past_key_features(
    key: pd.Series, t: pd.Series, amt: pd.Series, prefix: str
) -> pd.DataFrame:
    """Per-row aggregates over strictly-earlier rows with the same key.

    key: group key (NaN -> all features NaN); t: integer seconds; amt: transaction amount.
    """
    n = len(key)
    valid = key.notna().to_numpy()
    codes = np.full(n, -1, dtype=np.int64)
    codes[valid] = pd.factorize(key[valid])[0]
    tt = t.to_numpy(dtype=np.int64)
    if tt.max() >= BIG:
        raise ValueError("time exceeds BIG; increase BIG")
    a = amt.to_numpy(dtype=np.float64)

    idx = np.flatnonzero(valid)
    comp = codes[idx] * BIG + tt[idx]
    order = np.argsort(comp, kind="stable")
    comp_s, a_s, t_s = comp[order], a[idx][order], tt[idx][order]
    # Prefix sums restart at every key, so a row's sums depend only on earlier rows of its own
    # key (bit-exact under truncation; a global cumsum would let other keys change rounding).
    code_s = codes[idx][order]
    c1 = pd.Series(a_s).groupby(code_s).cumsum().to_numpy()
    c2 = pd.Series(a_s * a_s).groupby(code_s).cumsum().to_numpy()

    q = comp  # query value for each valid row: its own (code, time)
    hi = np.searchsorted(comp_s, q, side="left")  # first row of same key with time >= t
    gstart = np.searchsorted(comp_s, codes[idx] * BIG, side="left")
    cnt = (hi - gstart).astype(np.float64)

    def pref(c: np.ndarray, i: np.ndarray) -> np.ndarray:
        """Sum of key rows [gstart, i) = within-key cumsum at i-1, or 0 if empty."""
        return np.where(i > gstart, c[np.maximum(i - 1, 0)], 0.0)

    out = {}
    for name, w in WINDOWS_S.items():
        lo = np.maximum(np.searchsorted(comp_s, q - w, side="left"), gstart)  # time >= t - w
        out[f"{prefix}_cnt_{name}"] = (hi - lo).astype(np.float64)
        out[f"{prefix}_amt_sum_{name}"] = pref(c1, hi) - pref(c1, lo)

    s1 = pref(c1, hi)
    s2 = pref(c2, hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(cnt > 0, s1 / cnt, np.nan)
        var = np.where(cnt > 1, (s2 - cnt * mean**2) / (cnt - 1), np.nan)
        std = np.sqrt(np.clip(var, 0, None))
        a_now = a[idx]
        out[f"{prefix}_n_prev"] = cnt
        out[f"{prefix}_amt_mean_prev"] = mean
        out[f"{prefix}_amt_std_prev"] = std
        out[f"{prefix}_amt_ratio"] = a_now / mean
        out[f"{prefix}_amt_z"] = np.where(std > 0, (a_now - mean) / std, np.nan)
    last_t = np.where(cnt > 0, t_s[np.maximum(hi - 1, 0)], -1)
    out[f"{prefix}_secs_since_last"] = np.where(cnt > 0, tt[idx] - last_t, np.nan)

    res = pd.DataFrame(np.nan, index=key.index, columns=list(out), dtype=np.float64)
    for c, v in out.items():
        res.iloc[idx, res.columns.get_loc(c)] = v
    return res


def past_means(key: pd.Series, t: pd.Series, cols: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Per-row mean of each column over strictly-earlier rows with the same key (non-null only).

    Same exactness guarantees as past_key_features (within-key prefix sums).
    """
    n = len(key)
    valid = key.notna().to_numpy()
    codes = np.full(n, -1, dtype=np.int64)
    codes[valid] = pd.factorize(key[valid])[0]
    tt = t.to_numpy(dtype=np.int64)
    idx = np.flatnonzero(valid)
    comp = codes[idx] * BIG + tt[idx]
    order = np.argsort(comp, kind="stable")
    comp_s, code_s = comp[order], codes[idx][order]
    hi = np.searchsorted(comp_s, comp, side="left")
    gstart = np.searchsorted(comp_s, codes[idx] * BIG, side="left")

    res = pd.DataFrame(np.nan, index=key.index,
                       columns=[f"{prefix}_{c}_mean_prev" for c in cols.columns])
    for j, c in enumerate(cols.columns):
        v = cols[c].to_numpy(dtype=np.float64)[idx][order]
        nn = ~np.isnan(v)
        cs = pd.Series(np.where(nn, v, 0.0)).groupby(code_s).cumsum().to_numpy()
        cn = pd.Series(nn.astype(np.float64)).groupby(code_s).cumsum().to_numpy()
        has = hi > gstart
        s = np.where(has, cs[np.maximum(hi - 1, 0)], 0.0)
        k = np.where(has, cn[np.maximum(hi - 1, 0)], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            res.iloc[idx, j] = np.where(k > 0, s / k, np.nan)
    return res


D_COLS = ["D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15"]  # "days since X" style columns
UID_MEAN_COLS = ["C1", "C5", "C13", "C14", "dist1", "D2n", "D4n", "D10n", "D15n", "V258", "V294"]
FREQ_KEYS = ["card2", "card5", "addr1", "R_emaildomain", "DeviceInfo", "id_30", "id_31", "id_33"]


def build_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """v1 + D-normalisation, amount cents, uid past means, past-only frequency counts."""
    out = build_features(df)
    day = np.floor(df["TransactionDT"] / 86400)
    new = {}
    for c in D_COLS:
        if c in df:
            new[f"{c}n"] = (day - df[c]).astype(np.float32)  # fixed date per client-event
    amt = df["TransactionAmt"].astype(np.float64)
    new["amt_cents"] = (amt - np.floor(amt)).round(3)
    new["amt_log"] = np.log1p(amt)
    out = pd.concat([out, pd.DataFrame(new, index=df.index)], axis=1)

    uid = add_uid(df)
    mcols = [c for c in UID_MEAN_COLS if c in out]
    parts = [past_means(uid, df["TransactionDT"], out[mcols], "uid")]
    for k in FREQ_KEYS:
        if k in df:
            f = past_key_features(df[k], df["TransactionDT"], df["TransactionAmt"], f"fq_{k}")
            parts.append(f[[f"fq_{k}_cnt_7d", f"fq_{k}_cnt_30d"]])
    return pd.concat([out] + parts, axis=1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df (time-sorted IEEE) with engineered point-in-time columns appended."""
    t = df["TransactionDT"]
    amt = df["TransactionAmt"]
    uid = add_uid(df)
    parts = [
        past_key_features(df["card1"], t, amt, "card1"),
        past_key_features(uid, t, amt, "uid"),
        past_key_features(df["P_emaildomain"], t, amt, "pemail"),
    ]
    out = pd.concat([df] + parts, axis=1)
    out["uid_known"] = uid.notna().astype(np.int8)
    return out
