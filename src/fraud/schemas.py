"""pandera schemas for the key columns of each dataset (other columns pass through)."""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

IEEE_SCHEMA = pa.DataFrameSchema(
    {
        "TransactionID": pa.Column(int, unique=True),
        "TransactionDT": pa.Column(int, pa.Check.ge(0)),
        "isFraud": pa.Column(int, pa.Check.isin([0, 1])),
        "TransactionAmt": pa.Column("float32", pa.Check.gt(0)),
        "ProductCD": pa.Column(str, pa.Check.isin(list("WCRHS"))),
        "card1": pa.Column(int, nullable=False),
        "t_days": pa.Column(float, pa.Check.ge(0)),
    },
    checks=pa.Check(lambda d: d["TransactionDT"].is_monotonic_increasing, error="not time-sorted"),
    strict=False,
    coerce=False,
)

LC_SCHEMA = pa.DataFrameSchema(
    {
        "id": pa.Column(int, unique=True),
        "issue_d": pa.Column("datetime64[ns]"),
        "Default": pa.Column(int, pa.Check.isin([0, 1])),
        "loan_amnt": pa.Column(int, pa.Check.gt(0)),
        "fico_n": pa.Column("float32", pa.Check.in_range(300, 850), nullable=True),
        "t_days": pa.Column(float, pa.Check.ge(0)),
    },
    checks=pa.Check(lambda d: d["issue_d"].is_monotonic_increasing, error="not time-sorted"),
    strict=False,
    coerce=False,
)


def validate_ieee(df: pd.DataFrame) -> pd.DataFrame:
    return IEEE_SCHEMA.validate(df, lazy=True)


def validate_lc(df: pd.DataFrame) -> pd.DataFrame:
    return LC_SCHEMA.validate(df, lazy=True)
