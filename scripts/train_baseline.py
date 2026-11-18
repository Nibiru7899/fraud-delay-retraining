"""Phase-1 baselines on raw IEEE-CIS features, time split. Logs to MLflow, prints a short table.

Usage: python scripts/train_baseline.py [--models lgb,lr]
"""
from __future__ import annotations

import argparse
import time

import mlflow
import pandas as pd

from fraud.config import ROOT, load_config
from fraud.metrics import score_all
from fraud.models import make_lr, predict_lgb, train_lgb
from fraud.splits import check_disjoint_ordered, time_window
from fraud.tracking import start_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lgb,lr")
    ap.add_argument("--features", choices=["raw", "eng"], default="raw")
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    ds = cfg["datasets"]["ieee"]
    b, hs = ds["baseline"], ds["heldout_start"]
    check_disjoint_ordered([tuple(b["train"]), tuple(b["valid"]), tuple(b["test"])])

    df = pd.read_parquet(ROOT / (ds["feat_path"] if args.features == "eng" else ds["path"]))
    parts = {k: time_window(df, *b[k], heldout_start=hs) for k in ("train", "valid", "test")}
    del df
    feats = [
        c for c in parts["train"].columns
        if c not in ds["drop_cols"]
        and not any(c.endswith(sfx) for sfx in ds.get("drop_feature_suffixes", []))
    ]
    X = {k: v[feats] for k, v in parts.items()}
    y = {k: v[ds["label"]].to_numpy() for k, v in parts.items()}

    rows = []
    for model in args.models.split(","):
        t0 = time.time()
        with start_run(f"baseline_{model}_{args.features}", "ieee", seed):
            mlflow.log_params({"model": model, "features": args.features, "n_features": len(feats),
                               **{f"win_{k}": str(b[k]) for k in b}})
            if model == "lgb":
                booster, enc = train_lgb(X["train"], y["train"], X["valid"], y["valid"], seed)
                mlflow.log_param("best_iteration", booster.best_iteration)
                preds = {k: predict_lgb(booster, enc, X[k]) for k in ("valid", "test")}
            elif model == "lr":
                pipe = make_lr(X["train"], seed).fit(X["train"], y["train"])
                preds = {k: pipe.predict_proba(X[k])[:, 1] for k in ("valid", "test")}
            else:
                raise ValueError(model)
            for k, p in preds.items():
                m = score_all(y[k], p)
                mlflow.log_metrics({f"{k}_{n}": v for n, v in m.items()})
                rows.append({"model": model, "split": k, **{n: round(v, 4) for n, v in m.items()}})
            mlflow.log_metric("train_seconds", time.time() - t0)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
