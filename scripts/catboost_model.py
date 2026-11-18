"""CatBoost candidate for the LightGBM+CatBoost ensemble (same windows + feature set as LightGBM).

Trains on TRAIN, early-stops on VALID, saves VALID and TEST predictions. Prints VALID metrics only;
the test window is scored once later, never here.

Usage: python scripts/catboost_model.py [--exclude reports/excl_final.txt]
"""
from __future__ import annotations

import argparse
import time

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError, Pool

from fraud.config import ROOT
from fraud.dataset import load_xy
from fraud.metrics import score_all
from fraud.tracking import start_run


def prep(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
    return X


def fit(task_type: str, seed: int, train: Pool, valid: Pool) -> CatBoostClassifier:
    params = {"iterations": 5000, "learning_rate": 0.05, "depth": 8, "eval_metric": "PRAUC",
              "od_type": "Iter", "od_wait": 200, "random_seed": seed, "border_count": 254,
              "task_type": task_type, "verbose": 500}
    params |= {"devices": "0"} if task_type == "GPU" else {"thread_count": 10}
    model = CatBoostClassifier(**params)
    model.fit(train, eval_set=valid, use_best_model=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="reports/excl_final.txt")
    a = ap.parse_args()
    exclude = (ROOT / a.exclude).read_text().split()
    X, y, feats, seed = load_xy("v2", True, exclude, windows=("train", "valid", "test"))
    cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(X["train"][c])]
    X = {k: prep(v, cat_cols) for k, v in X.items()}
    pools = {k: Pool(v, y[k], cat_features=cat_cols) for k, v in X.items()}
    t0 = time.time()
    task_type = "GPU"
    try:
        model = fit("GPU", seed, pools["train"], pools["valid"])
    except CatBoostError as e:
        print(f"GPU training failed ({str(e)[:120]}); falling back to CPU")
        task_type = "CPU"
        model = fit("CPU", seed, pools["train"], pools["valid"])
    secs = time.time() - t0
    preds = {k: model.predict_proba(pools[k])[:, 1].astype(np.float64) for k in ("valid", "test")}
    for k, p in preds.items():
        np.save(ROOT / f"reports/preds_cat_{k}.npy", p)
    m = score_all(y["valid"], preds["valid"])
    with start_run("catboost_gpu", "ieee", seed):
        mlflow.set_tag("stage", "ensemble-candidate")
        mlflow.log_params({"feat": "v2", "drop_raw_d": True, "n_excluded": len(exclude),
                           "n_features": len(feats), "n_cat": len(cat_cols), "task_type": task_type,
                           "best_iteration": model.get_best_iteration(), "depth": 8,
                           "learning_rate": 0.05, "border_count": 254})
        mlflow.log_metrics({f"valid_{k}": v for k, v in m.items()} | {"train_secs": secs})
    print(f"catboost {task_type} nfeat={len(feats)} ncat={len(cat_cols)} "
          f"best_iter={model.get_best_iteration()} VALID pr_auc={m['pr_auc']:.4f} "
          f"roc_auc={m['roc_auc']:.4f} rec@1%={m['recall_at_1pct_fpr']:.4f} secs={secs:.0f}")


if __name__ == "__main__":
    main()
