"""Week 4: imbalance handling x threshold tuning on IEEE (engineered features).

Variants (all LightGBM, same fixed params, seed): none | class_weight | smote (SMOTENC, train only).
Choice rule fixed in advance: pick by VALID PR-AUC; thresholds are chosen on VALID by max net value
and then applied unchanged to TEST. SMOTE rows exist only inside training; all evaluation is on
real rows.
"""
from __future__ import annotations

import time

import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

from fraud.config import ROOT, load_config
from fraud.cost import best_threshold, net_value
from fraud.metrics import score_all
from fraud.models import CatEncoder, predict_lgb, split_cols, train_lgb
from fraud.splits import time_window
from fraud.tracking import start_run

SMOTE_RATIO = 0.1  # minority:majority after resampling (fixed a priori)


def smote_train(Xtr: pd.DataFrame, ytr: np.ndarray, seed: int):
    """Impute (train medians), ordinal-code categoricals, SMOTENC. Returns X, y, transform fn."""
    num, cat = split_cols(Xtr)
    med = Xtr[num].median()
    enc = CatEncoder().fit(Xtr, cat)

    def prep(X: pd.DataFrame) -> pd.DataFrame:
        X = enc.transform(X)
        out = X[num].fillna(med).astype(np.float32)
        for c in cat:
            out[c] = X[c].cat.codes.astype(np.int32)  # unseen/NaN -> -1
        return out[num + cat]

    Xp = prep(Xtr)
    cat_idx = [Xp.columns.get_loc(c) for c in cat]
    sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=SMOTE_RATIO, random_state=seed)
    Xr, yr = sm.fit_resample(Xp, ytr)
    Xr = pd.DataFrame(Xr, columns=Xp.columns)
    for c in cat:  # back to categorical so LightGBM treats codes as categories
        Xr[c] = Xr[c].astype(np.int32).astype(str)
    return Xr, yr, prep


def main() -> None:
    cfg = load_config()
    seed, ds = cfg["seed"], cfg["datasets"]["ieee"]
    costs = cfg["review_cost_usd"]
    b, hs = ds["baseline"], ds["heldout_start"]
    df = pd.read_parquet(ROOT / ds["feat_path"])
    parts = {k: time_window(df, *b[k], heldout_start=hs) for k in ("train", "valid", "test")}
    del df
    sfx = tuple(ds["drop_feature_suffixes"])
    feats = [c for c in parts["train"].columns if c not in ds["drop_cols"] and not c.endswith(sfx)]
    X = {k: v[feats] for k, v in parts.items()}
    y = {k: v[ds["label"]].to_numpy() for k, v in parts.items()}
    amt = {k: v["TransactionAmt"].to_numpy(np.float64) for k, v in parts.items()}
    spw = float((y["train"] == 0).sum() / (y["train"] == 1).sum())

    rows = []
    for variant in ["none", "class_weight", "smote"]:
        t0 = time.time()
        with start_run(f"imbalance_{variant}", "ieee", seed):
            mlflow.log_params({"variant": variant, "features": "eng", "n_features": len(feats)})
            if variant == "smote":
                Xr, yr, prep = smote_train(X["train"], y["train"], seed)
                Xv = prep(X["valid"])
                for c in split_cols(X["train"])[1]:
                    Xv[c] = Xv[c].astype(str)
                booster, enc = train_lgb(Xr, yr, Xv, y["valid"], seed)

                def pred(k, booster=booster, enc=enc, prep=prep):
                    Z = prep(X[k])
                    for c in split_cols(X["train"])[1]:
                        Z[c] = Z[c].astype(str)
                    return predict_lgb(booster, enc, Z)
            else:
                extra = {"scale_pos_weight": spw} if variant == "class_weight" else {}
                booster, enc = train_lgb(X["train"], y["train"], X["valid"], y["valid"], seed,
                                         extra_params=extra)

                def pred(k, booster=booster, enc=enc):
                    return predict_lgb(booster, enc, X[k])
            p = {k: pred(k) for k in ("valid", "test")}
            mlflow.log_param("best_iteration", booster.best_iteration)
            for k in ("valid", "test"):
                m = score_all(y[k], p[k])
                mlflow.log_metrics({f"{k}_{n}": v for n, v in m.items()})
            row = {"variant": variant,
                   "val_pr_auc": score_all(y["valid"], p["valid"])["pr_auc"],
                   "test_pr_auc": score_all(y["test"], p["test"])["pr_auc"],
                   "test_roc_auc": score_all(y["test"], p["test"])["roc_auc"],
                   "test_rec@1%": score_all(y["test"], p["test"])["recall_at_1pct_fpr"]}
            for c in costs:
                thr, vval = best_threshold(y["valid"], p["valid"], amt["valid"], c)
                tval = net_value(y["test"], p["test"], amt["test"], thr, c)
                _, toracle = best_threshold(y["test"], p["test"], amt["test"], c)
                mlflow.log_metrics({f"thr_c{c}": thr, f"valid_net_c{c}": vval,
                                    f"test_net_c{c}": tval, f"test_net_oracle_c{c}": toracle})
                row[f"test_net_c{c}"] = tval
                row[f"oracle_c{c}"] = toracle
            mlflow.log_metric("train_seconds", time.time() - t0)
            rows.append(row)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(ROOT / "reports/imbalance_comparison.csv", index=False)
    print(out.to_string(index=False))
    print("caught-fraud max on test (sum amt of frauds):", round(amt["test"][y["test"] == 1].sum()))


if __name__ == "__main__":
    main()
