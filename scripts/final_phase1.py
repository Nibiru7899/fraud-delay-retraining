"""Phase 1 final model: fit LightGBM (seed-averaged), choose LGB+CatBoost blend weight on VALID,
then score the test window exactly once.

  python scripts/final_phase1.py fit   --params configs/lgb_tuned.json --seeds 42 1 2 --tag tuned
  python scripts/final_phase1.py final --tag tuned      # writes reports/phase1_final.json

Valid/test are both inside the dev period [0, 154); the held-out tail stays locked.
"""
from __future__ import annotations

import argparse
import json

import mlflow
import numpy as np

from fraud.config import ROOT, load_config
from fraud.cost import best_threshold, net_value
from fraud.dataset import load_xy
from fraud.metrics import score_all
from fraud.models import predict_lgb, train_lgb
from fraud.tracking import start_run

EXCL = "configs/excl_final.txt"
WINDOWS = ("train", "valid", "test")
MIN_GAIN = 0.005  # adoption rule on VALID PR-AUC


def load():
    return load_xy("v2", True, (ROOT / EXCL).read_text().split(), windows=WINDOWS)


def fit(a) -> None:
    params = json.loads((ROOT / a.params).read_text())["params"] if a.params else {}
    X, y, feats, _ = load()
    preds = {k: [] for k in ("valid", "test")}
    with start_run(f"final_fit_{a.tag}", "ieee", a.seeds[0]):
        mlflow.log_params({"seeds": str(a.seeds), "n_features": len(feats),
                           "extra_params": json.dumps(params)})
        for s in a.seeds:
            bst, enc = train_lgb(X["train"], y["train"], X["valid"], y["valid"], s,
                                 extra_params=params)
            for k, lst in preds.items():
                lst.append(predict_lgb(bst, enc, X[k]))
            v = score_all(y["valid"], preds["valid"][-1])["pr_auc"]
            print(f"seed {s} iters {bst.best_iteration} VALID pr_auc {v:.4f}")
        for k, p in preds.items():
            np.save(ROOT / f"reports/preds_lgb_{a.tag}_{k}.npy", np.mean(p, axis=0))
        v = score_all(y["valid"], np.mean(preds["valid"], axis=0))["pr_auc"]
        mlflow.log_metric("valid_pr_auc_seedavg", v)
    print(f"seed-avg VALID pr_auc {v:.4f}")


def final(a) -> None:
    cfg = load_config()
    X, y, _, _ = load()
    amt = {k: X[k]["TransactionAmt"].to_numpy() for k in ("valid", "test")}
    lgb = {k: np.load(ROOT / f"reports/preds_lgb_{a.tag}_{k}.npy") for k in ("valid", "test")}
    cat = {k: np.load(ROOT / f"reports/preds_cat_{k}.npy") for k in ("valid", "test")}
    assert all(len(lgb[k]) == len(cat[k]) == len(y[k]) for k in lgb)

    # blend weight on CatBoost chosen on VALID only
    grid = np.round(np.arange(0, 1.01, 0.1), 2)
    vs = {w: score_all(y["valid"], (1 - w) * lgb["valid"] + w * cat["valid"])["pr_auc"]
          for w in grid}
    w_best = max(vs, key=vs.get)
    w = w_best if vs[w_best] - vs[0.0] >= MIN_GAIN else 0.0
    print("valid pr_auc by cat weight:", {k: round(v, 4) for k, v in vs.items()})
    print(f"chosen cat weight {w} (best {w_best}, gain {vs[w_best] - vs[0.0]:+.4f})")

    p = {k: (1 - w) * lgb[k] + w * cat[k] for k in ("valid", "test")}
    out = {"tag": a.tag, "cat_weight": float(w), "valid": score_all(y["valid"], p["valid"]),
           "test": score_all(y["test"], p["test"]), "net_value": {}}
    for rc in cfg["review_cost_usd"]:
        thr, _ = best_threshold(y["valid"], p["valid"], amt["valid"], rc)
        _, oracle = best_threshold(y["test"], p["test"], amt["test"], rc)
        out["net_value"][str(rc)] = {
            "threshold_from_valid": float(thr),
            "test_net": float(net_value(y["test"], p["test"], amt["test"], thr, rc)),
            "test_oracle": float(oracle),
            "test_max_catchable": float(amt["test"][y["test"] == 1].sum())}
    with start_run(f"final_test_{a.tag}", "ieee", cfg["seed"]):
        mlflow.set_tag("stage", "phase1-final-test")
        mlflow.log_param("cat_weight", w)
        mlflow.log_metrics({f"{s}_{k}": v for s in ("valid", "test") for k, v in out[s].items()})
    (ROOT / "reports/phase1_final.json").write_text(json.dumps(out, indent=2))
    t = out["test"]
    print(f"TEST pr_auc {t['pr_auc']:.4f} roc_auc {t['roc_auc']:.4f} "
          f"rec@1% {t['recall_at_1pct_fpr']:.4f}")
    for rc, d in out["net_value"].items():
        print(f"  review ${rc}: net {d['test_net']:,.0f} oracle {d['test_oracle']:,.0f} "
              f"max {d['test_max_catchable']:,.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "final"])
    ap.add_argument("--params", default="")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--tag", default="default")
    a = ap.parse_args()
    fit(a) if a.mode == "fit" else final(a)


if __name__ == "__main__":
    main()
