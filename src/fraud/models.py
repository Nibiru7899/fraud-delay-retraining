"""Baseline models. Every encoder is fit on the training window only."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_cols(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    return num, cat


# ---------- LightGBM ----------
LGB_PARAMS = {  # fixed a priori, not tuned on test
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "metric": "average_precision",
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 8,
    "verbose": -1,
}


class CatEncoder:
    """String columns -> pandas category with categories frozen from train; unseen -> NaN."""

    def fit(self, X: pd.DataFrame, cat_cols: list[str]) -> CatEncoder:
        self.cats = {c: pd.Index(X[c].dropna().unique()).sort_values() for c in cat_cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, cats in self.cats.items():
            X[c] = pd.Categorical(X[c], categories=cats)
        return X


def train_lgb(
    Xtr, ytr, Xva, yva, seed: int, max_rounds: int = 5000, patience: int = 200,
    extra_params: dict | None = None,
):
    _, cat = split_cols(Xtr)
    enc = CatEncoder().fit(Xtr, cat)
    dtr = lgb.Dataset(enc.transform(Xtr), ytr, categorical_feature=cat or "auto")
    dva = lgb.Dataset(enc.transform(Xva), yva, reference=dtr)
    booster = lgb.train(
        {**LGB_PARAMS, "seed": seed, **(extra_params or {})},
        dtr,
        num_boost_round=max_rounds,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    return booster, enc


def predict_lgb(booster, enc, X) -> np.ndarray:
    return booster.predict(enc.transform(X), num_iteration=booster.best_iteration)


# ---------- Logistic regression ----------
def make_lr(X: pd.DataFrame, seed: int) -> Pipeline:
    num, cat = split_cols(X)
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="NA")),
                              ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                   min_frequency=100, max_categories=30))]), cat),
        ]
    )
    return Pipeline([("pre", pre),
                     ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=seed))])
