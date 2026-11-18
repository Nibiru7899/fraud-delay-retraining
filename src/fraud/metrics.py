"""Evaluation metrics fixed in context.md (PR-AUC main; ROC-AUC for rule 5; recall at 1% FPR)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(y: np.ndarray, p: np.ndarray, max_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def score_all(y, p) -> dict[str, float]:
    y = np.asarray(y)
    p = np.asarray(p)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "recall_at_1pct_fpr": recall_at_fpr(y, p, 0.01),
        "pos_rate": float(y.mean()),
        "n": len(y),
    }
