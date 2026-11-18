"""Adversarial validation: can a model tell TRAIN-window rows from VALID-window rows?

High-importance features in that model encode time (drift / clocks). Writes the ranked list to
reports/adversarial_importance.csv. Uses no fraud labels.
Usage: python scripts/adversarial.py --feat v2 [--drop-raw-d]
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fraud.config import ROOT
from fraud.dataset import load_xy
from fraud.models import LGB_PARAMS, CatEncoder, split_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="v2")
    ap.add_argument("--drop-raw-d", action="store_true")
    ap.add_argument("--exclude", default="")
    a = ap.parse_args()
    exclude = (ROOT / a.exclude).read_text().split() if a.exclude else []
    X, _, _feats, seed = load_xy(a.feat, a.drop_raw_d, exclude)
    Z = pd.concat([X["train"], X["valid"]], ignore_index=True)
    is_valid = np.r_[np.zeros(len(X["train"])), np.ones(len(X["valid"]))]
    # random 50/50 split is fine HERE: the target is "which window", not fraud
    r = np.random.default_rng(seed)
    m = r.uniform(size=len(Z)) < 0.5
    _, cat = split_cols(Z)
    enc = CatEncoder().fit(Z[m], cat)
    params = {**LGB_PARAMS, "metric": "auc", "seed": seed, "learning_rate": 0.1}
    bst = lgb.train(params, lgb.Dataset(enc.transform(Z[m]), is_valid[m]), num_boost_round=200)
    auc = roc_auc_score(is_valid[~m], bst.predict(enc.transform(Z[~m])))
    imp = pd.Series(bst.feature_importance("gain"), index=bst.feature_name())
    imp = (imp / imp.sum()).sort_values(ascending=False)
    imp.to_csv(ROOT / "reports/adversarial_importance.csv", header=["gain_share"])
    print(f"adversarial AUC (train vs valid window): {auc:.4f}")
    print(imp.head(15).round(4).to_string())


if __name__ == "__main__":
    main()
