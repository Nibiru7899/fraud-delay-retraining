"""Raw-data loaders. Output is time-sorted, with a numeric `t_days` column (days since stream start).

Nothing here creates features or labels; it only reads, joins, types and orders real data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraud.config import ROOT

DATA = ROOT / "data"
IEEE_DIR = DATA / "ieee-fraud-detection"
LC_FILE = DATA / "LC_loans_granting_model_dataset.csv"
INTERIM = DATA / "interim"

IEEE_TIME = "TransactionDT"
IEEE_LABEL = "isFraud"
LC_TIME = "issue_d"
LC_LABEL = "Default"


def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.select_dtypes("float64").columns
    df[cols] = df[cols].astype(np.float32)
    return df


def load_ieee_raw(ieee_dir: Path = IEEE_DIR) -> pd.DataFrame:
    """Labelled IEEE-CIS train data: transaction LEFT JOIN identity, sorted by time."""
    tx = pd.read_csv(ieee_dir / "train_transaction.csv")
    idn = pd.read_csv(ieee_dir / "train_identity.csv")
    n = len(tx)
    df = tx.merge(idn, on="TransactionID", how="left", validate="one_to_one")
    assert len(df) == n, "identity join changed row count"
    df = _downcast_floats(df)
    # stable sort: ties keep original (TransactionID) order -> deterministic
    df = df.sort_values([IEEE_TIME, "TransactionID"], kind="stable").reset_index(drop=True)
    df["t_days"] = (df[IEEE_TIME] - df[IEEE_TIME].min()) / 86400.0
    return df


def load_lc_raw(lc_file: Path = LC_FILE) -> pd.DataFrame:
    """LendingClub granting dataset, sorted by issue month. `t_days` = days since first month."""
    df = pd.read_csv(lc_file, low_memory=False)
    df[LC_TIME] = pd.to_datetime(df[LC_TIME], format="%b-%Y")
    df = _downcast_floats(df)
    df = df.sort_values([LC_TIME, "id"], kind="stable").reset_index(drop=True)
    df["t_days"] = (df[LC_TIME] - df[LC_TIME].min()).dt.days.astype(np.float64)
    return df
