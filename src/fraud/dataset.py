"""Load the IEEE-CIS feature table and cut baseline windows (shared by LightGBM/CatBoost scripts)."""
from __future__ import annotations

import pandas as pd

from fraud.config import ROOT, load_config
from fraud.features import D_COLS
from fraud.splits import time_window

FEAT_PATHS = {"raw": "data/interim/ieee.parquet", "v1": "data/processed/ieee_feat.parquet",
              "v2": "data/processed/ieee_feat_v2.parquet"}


def load_xy(feat: str, drop_raw_d: bool, exclude: list[str],
            windows: tuple[str, ...] = ("train", "valid")):
    """X/y dicts keyed by baseline window name; feature list is derived from the train window."""
    cfg = load_config()
    ds = cfg["datasets"]["ieee"]
    b, hs = ds["baseline"], ds["heldout_start"]
    df = pd.read_parquet(ROOT / FEAT_PATHS[feat])
    parts = {k: time_window(df, *b[k], heldout_start=hs) for k in windows}
    sfx = tuple(ds["drop_feature_suffixes"])
    drop = set(ds["drop_cols"]) | set(exclude) | (set(D_COLS) if drop_raw_d else set())
    feats = [c for c in parts["train"].columns if c not in drop and not c.endswith(sfx)]
    X = {k: v[feats] for k, v in parts.items()}
    y = {k: v[ds["label"]].to_numpy() for k, v in parts.items()}
    return X, y, feats, cfg["seed"]
