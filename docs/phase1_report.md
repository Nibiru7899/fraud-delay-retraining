# Phase 1 report — baseline, features, model (IEEE-CIS)

All numbers are on real data. Windows are cut by time, in days since the first transaction:
train [0, 112), valid [112, 133), test [133, 154). The held-out tail [154, 182] has not been touched.
The test window was scored only for the baselines and once for the final model. Every choice was made on validation.

## Comparison table (test window 133–154)

| Model | PR-AUC | ROC-AUC | Recall @1% FPR | Net value @ $10/review |
|---|---|---|---|---|
| Logistic regression, raw | 0.3608 | 0.8326 | 0.2984 | — |
| LightGBM, raw features, fixed params | 0.5336 | 0.9011 | 0.4712 | — |
| LightGBM, v1 point-in-time features | 0.5390 | 0.9030 | — | $141.6k |
| **LightGBM, v2 features + tuned + 3-seed average (final)** | **0.5933** | **0.9247** | **0.5283** | **$160.6k** |

The final model's PR-AUC is +0.060 over the raw-feature baseline (+11% relative). Recall at 1% FPR is +0.057.

Net value on the test window uses thresholds chosen on validation:

| Review cost | Net | Best possible threshold (test) | Share of best |
|---|---|---|---|
| $5 | $197.0k | $202.7k | 97% |
| $10 | $160.6k | $167.0k | 96% |
| $20 | $113.1k | $116.1k | 97% |

The most fraud money that can be caught on the test window is $288.0k.

Rule 5 passes: the raw-feature baseline's ROC-AUC of 0.9011 matches the published temporal AUC of 0.9004.

## Validation iteration log (adopt a change only if the gain is at least 0.005)

| Step | VALID PR-AUC | Decision |
|---|---|---|
| Raw-feature baseline | 0.6759 | — |
| v1 features (card1/uid/email velocity, amount deviation) | 0.6792 | kept (basis for v2) |
| Imbalance handling: class_weight 0.6765 / SMOTENC 0.6824 | below bar | kept "none" (better calibrated) |
| v2 features (D-normalised, uid past means, past-only frequencies) | 0.6883 | adopted |
| Drop raw D columns | 0.6905 | adopted (removes a time proxy) |
| Drop pemail_* and secs-since-last (time proxies) | 0.699 (3-seed mean) | adopted |
| Optuna tuning (25 trials, 2 time folds) | 0.712 (2-seed mean) | adopted |
| Average of 3 seeds | 0.7142 | adopted |
| CatBoost (GPU) alone | 0.6551 | — |
| LightGBM + CatBoost blend, probability or rank | always worse | rejected |

## Findings to carry into Phase 2

1. **Decay is real.** PR-AUC drops from 0.714 on validation to 0.593 on test, three weeks later.
   - Some of that drop is optimism. Early stopping and tuning fold B both used the validation window.
   - The rest is drift, and drift is the effect the retraining study is meant to measure.
2. **Time proxies.** Adversarial validation separates train from valid with AUC 1.0.
   - The strongest separators are the D-normalised features and the email/recency aggregates.
   - Removing the recency aggregates helped. Removing the D-normalised features hurt badly (0.662).
   - So some time-correlated features carry real signal. The drift monitor has to account for this.
3. **Calibration.** Mean predicted probability on validation is 0.023, against an actual fraud rate of 0.041.
   - The model underestimates fraud under prior shift.
   - CBPE needs calibrated probabilities, so open decision 6 (CBPE calibration) is a real issue.
4. **LendingClub censoring.** The stream is cut to 2010–2015 because defaults collapse after 2016Q2.
   - Some 5-year loans in 2014–2015 may still be partly censored. There is no term column to check this.
5. **Reproducibility.** A rerun of v1 reproduced validation PR-AUC 0.6792 exactly.
   - The leak tests pass with exact equality on both handmade and real data.

## Artifacts

- Code
  - `scripts/final_phase1.py` (fit and final scoring)
  - `scripts/tune.py`
  - `scripts/eval_valid.py`
  - `scripts/catboost_model.py`
- Settings: `configs/lgb_tuned.json` and `configs/excl_final.txt` (copied from `reports/`, so they are tracked).
- Run tracking: MLflow experiment `fraud-delay-retrain` in `mlruns.db`.
  - Every run records the git commit, the data hash and the seed.
