import numpy as np
import pandas as pd
import pandera.errors as pe
import pytest

from fraud.schemas import validate_ieee, validate_lc


def _ieee():
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [86400, 86401, 90000],
            "isFraud": [0, 1, 0],
            "TransactionAmt": np.array([10.5, 20.0, 3.25], dtype="float32"),
            "ProductCD": ["W", "C", "H"],
            "card1": [1000, 2000, 1000],
            "t_days": [0.0, 1 / 86400, 3600 / 86400],
        }
    )


def _lc():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "issue_d": pd.to_datetime(["2010-01-01", "2010-02-01"]).astype("datetime64[ns]"),
            "Default": [0, 1],
            "loan_amnt": [1000, 5000],
            "fico_n": np.array([700, np.nan], dtype="float32"),
            "t_days": [0.0, 31.0],
        }
    )


def test_valid_frames_pass():
    validate_ieee(_ieee())
    validate_lc(_lc())


@pytest.mark.parametrize(
    "col,val",
    [("isFraud", 2), ("TransactionAmt", np.float32(-1)), ("ProductCD", "X")],
)
def test_ieee_bad_values_fail(col, val):
    df = _ieee()
    df.loc[0, col] = val
    with pytest.raises(pe.SchemaErrors):
        validate_ieee(df)


def test_ieee_unsorted_fails():
    df = _ieee().iloc[::-1].reset_index(drop=True)
    with pytest.raises(pe.SchemaErrors):
        validate_ieee(df)


def test_ieee_duplicate_id_fails():
    df = _ieee()
    df.loc[1, "TransactionID"] = 1
    with pytest.raises(pe.SchemaErrors):
        validate_ieee(df)


def test_lc_bad_fico_fails():
    df = _lc()
    df.loc[0, "fico_n"] = np.float32(900)
    with pytest.raises(pe.SchemaErrors):
        validate_lc(df)
