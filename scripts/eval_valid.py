"""Iteration harness: train LightGBM on train window, score VALID only (test never computed here).

Usage:
  python scripts/eval_valid.py --feat v2 [--drop-raw-d] [--exclude reports/drop_features.txt]
                               [--params '{"num_leaves": 255}'] [--tag name]
"""
from __future__ import annotations

import argparse
import json
import time

import mlflow

from fraud.config import ROOT
from fraud.dataset import FEAT_PATHS, load_xy
from fraud.metrics import score_all
from fraud.models import train_lgb
from fraud.tracking import start_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="v2", choices=list(FEAT_PATHS))
    ap.add_argument("--drop-raw-d", action="store_true")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    exclude = (ROOT / a.exclude).read_text().split() if a.exclude else []
    params = json.loads(a.params)
    X, y, feats, seed = load_xy(a.feat, a.drop_raw_d, exclude)
    seed = a.seed if a.seed is not None else seed
    t0 = time.time()
    with start_run(f"iter_{a.feat}_{a.tag}", "ieee", seed):
        mlflow.set_tag("stage", "iterate-valid-only")
        mlflow.log_params({"feat": a.feat, "drop_raw_d": a.drop_raw_d, "n_excluded": len(exclude),
                           "n_features": len(feats), "extra_params": json.dumps(params)})
        booster, enc = train_lgb(X["train"], y["train"], X["valid"], y["valid"], seed,
                                 extra_params=params)
        from fraud.models import predict_lgb
        m = score_all(y["valid"], predict_lgb(booster, enc, X["valid"]))
        mlflow.log_metrics({f"valid_{k}": v for k, v in m.items()})
        mlflow.log_param("best_iteration", booster.best_iteration)
    print(f"{a.feat} tag={a.tag} nfeat={len(feats)} iters={booster.best_iteration} "
          f"VALID pr_auc={m['pr_auc']:.4f} roc_auc={m['roc_auc']:.4f} "
          f"rec@1%={m['recall_at_1pct_fpr']:.4f} secs={time.time() - t0:.0f}")


if __name__ == "__main__":
    main()
