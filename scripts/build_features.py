"""Build point-in-time features on IEEE-CIS -> data/processed/ieee_feat[_v2].parquet.

Usage: python scripts/build_features.py [v1|v2]
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from fraud.config import ROOT
from fraud.features import build_features, build_features_v2


def main() -> None:
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data/interim/ieee.parquet")
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    out = build_features_v2(df) if ver == "v2" else build_features(df)
    new = [c for c in out.columns if c not in df.columns]
    out.to_parquet(ROOT / f"data/processed/ieee_feat{'_v2' if ver == 'v2' else ''}.parquet", index=False)
    print(f"rows {len(out)}  new features {len(new)}  secs {time.time() - t0:.1f}")
    print(out[new].describe().T[["count", "mean", "50%", "max"]].round(2).tail(30).to_string())


if __name__ == "__main__":
    main()
