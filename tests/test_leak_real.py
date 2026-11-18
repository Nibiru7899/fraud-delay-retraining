"""No-future-leak test on REAL IEEE rows (skipped when data is absent, e.g. in CI)."""
import pandas as pd
import pytest

from fraud.config import ROOT
from fraud.features import build_features_v2

P = ROOT / "data/interim/ieee.parquet"
COLS = None  # all columns: v2 uses many


@pytest.mark.skipif(not P.exists(), reason="IEEE parquet not present")
@pytest.mark.parametrize("cut_day", [30, 90])
def test_no_future_leak_real(cut_day):
    df = pd.read_parquet(P, columns=COLS)
    cut = df["TransactionDT"].min() + cut_day * 86400
    trunc_df = df[df["TransactionDT"] < cut]
    full = build_features_v2(df).loc[trunc_df.index]
    trunc = build_features_v2(trunc_df)
    sample = trunc.sample(n=min(20000, len(trunc)), random_state=0).index
    pd.testing.assert_frame_equal(full.loc[sample], trunc.loc[sample], check_exact=True)
