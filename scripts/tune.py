"""Optuna tuning of LightGBM on two rolling time folds inside the dev period (test never used).

Fold A: train [0, 91)  -> valid [91, 112)
Fold B: train [0, 112) -> valid [112, 133)
Objective: mean PR-AUC over folds. Best params -> reports/best_params.json.
Usage: python scripts/tune.py --feat v2 [--drop-raw-d] [--exclude file] --trials 30
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import optuna
import pandas as pd

from fraud.config import ROOT, load_config
from fraud.features import D_COLS
from fraud.metrics import score_all
from fraud.models import predict_lgb, train_lgb
from fraud.splits import time_window

FOLDS = [((0, 91), (91, 112)), ((0, 112), (112, 133))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="v2")
    ap.add_argument("--drop-raw-d", action="store_true")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    cfg = load_config()
    ds, seed = cfg["datasets"]["ieee"], cfg["seed"]
    path = {"v1": "data/processed/ieee_feat.parquet",
            "v2": "data/processed/ieee_feat_v2.parquet"}[a.feat]
    df = pd.read_parquet(ROOT / path, )
    df = time_window(df, 0, 133, heldout_start=ds["heldout_start"])  # dev minus test window
    exclude = set((ROOT / a.exclude).read_text().split()) if a.exclude else set()
    drop = set(ds["drop_cols"]) | exclude | (set(D_COLS) if a.drop_raw_d else set())
    sfx = tuple(ds["drop_feature_suffixes"])
    feats = [c for c in df.columns if c not in drop and not c.endswith(sfx)]
    folds = []
    for tr, va in FOLDS:
        a_, b_ = df[(df.t_days >= tr[0]) & (df.t_days < tr[1])], df[
            (df.t_days >= va[0]) & (df.t_days < va[1])]
        folds.append((a_[feats], a_[ds["label"]].to_numpy(), b_[feats], b_[ds["label"]].to_numpy()))
    del df

    def objective(trial: optuna.Trial) -> float:
        p = {
            "learning_rate": 0.1,  # faster search; final model refit at 0.05
            "num_leaves": trial.suggest_int("num_leaves", 31, 1023, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [63, 255]),
            "cat_smooth": trial.suggest_float("cat_smooth", 1, 100, log=True),
        }
        scores, iters = [], []
        for Xtr, ytr, Xva, yva in folds:
            bst, enc = train_lgb(Xtr, ytr, Xva, yva, seed, extra_params=p, patience=100)
            scores.append(score_all(yva, predict_lgb(bst, enc, Xva))["pr_auc"])
            iters.append(bst.best_iteration)
        trial.set_user_attr("fold_pr_auc", [round(s, 4) for s in scores])
        trial.set_user_attr("iters", iters)
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                storage=f"sqlite:///{(ROOT / 'reports/optuna.db').as_posix()}",
                                study_name=f"lgb_{a.feat}_{int(a.drop_raw_d)}_{len(exclude)}",
                                load_if_exists=True)
    # trial 0 = current defaults, so the search must beat them
    study.enqueue_trial({"num_leaves": 127, "min_child_samples": 100, "feature_fraction": 0.5,
                         "bagging_fraction": 0.8, "lambda_l2": 1.0, "lambda_l1": 0.001,
                         "max_bin": 255, "cat_smooth": 10.0})
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials)
    best = study.best_trial
    out = {"params": best.params, "mean_pr_auc": best.value, **best.user_attrs,
           "default_trial": study.trials[0].value, "n_trials": len(study.trials)}
    (ROOT / "reports/best_params.json").write_text(json.dumps(out, indent=2))
    print(f"default {study.trials[0].value:.4f} -> best {best.value:.4f} "
          f"folds {best.user_attrs['fold_pr_auc']} in {time.time() - t0:.0f}s")
    print(json.dumps(best.params))


if __name__ == "__main__":
    main()
