# context.md — project memory (read by section, not whole)

Sections: Goal · Datasets · Locked decisions · Open decisions · Phases · Accuracy rules · Stack · Status log

## Goal
Claim to fill (only via protocol): "Under a 30–60 day label delay, policy X kept PR-AUC within __
of the best policy using __% fewer retrains, 95% CI." Ties / losses are valid results.
Novel bit: GBDT + realistic chargeback delay + CBPE label-free trigger + hybrid policy.
Source plan: `Fraud detection with delay-aware retraining project plan.pdf` (don't re-read; this file distills it).
Calendar: W1 = 23 Nov 2026 … W12 ends 14 Feb 2027.

## Datasets (all real) — ON DISK as of 2026-11-18, provided by user
- `data/ieee-fraud-detection/`: train_transaction 652 MB, train_identity 26 MB (+ unlabeled test_*, unused). Train: 590,540 rows, fraud 3.50%, span 182.0 days, already time-sorted.
- `data/LC_loans_granting_model_dataset.csv`: 1,347,681 rows × 15 cols, issue_d Jun-2007 → Dec-2018 (139 months), Default rate 19.98%, no null labels. Columns: id, issue_d, revenue, dti_n, loan_amnt, fico_n, experience_c, emp_length, purpose, home_ownership_n, addr_state, zip_code, Default, title, desc. This is a pre-cleaned origination-only version (no post-loan leak columns) — NOT the wordsforthewise mirror. Licence/source still to confirm. `title`/`desc` are free text; `id` must not be a feature.

Original source notes:
| Name | Source | Time col | Notes |
|---|---|---|---|
| IEEE-CIS | `kaggle competitions download -c ieee-fraud-detection` (accept rules first) | `TransactionDT` (seconds offset, ~182 days in train) | Only `train_*` has labels → all splits from it. Join transaction+identity on `TransactionID`. |
| IEEE-CIS + injected drift | derived from IEEE-CIS | same | Semi-synthetic; report separately. Drift = documented transform at known dates. |
| LendingClub 2007–2018 | Kaggle mirror `wordsforthewise/lending-club` (verify licence W1) | `issue_d` (month) | Label = Charged Off/Default vs Fully Paid; drop `Current`/in-progress loans. Drop post-origination columns (payments, recoveries, last_*) — they leak. |
| Fallback 2nd stream | decide by end W1 if LendingClub fails | — | Must be real data with known drift. |

## Locked decisions (don't re-litigate)
- Main model LightGBM; baselines logistic regression + small PyTorch MLP.
- Main metric PR-AUC per future window, averaged over stream. Also ROC-AUC (for rule 5), recall@1% FPR, money caught − review cost, #retrains, compute minutes, detection delay (injected runs), |CBPE est − true| .
- Delays: 0, 7, 30, 60 days. Policies: (1) never, (2) fixed schedule, (3) drift alarm PSI/ADWIN, (4) arrived-label accuracy drop, (5) CBPE drop, (6) hybrid schedule + triggered top-ups. Same retrain budget N for all.
- ≥10 replays per setting (different start points + delay samples), 95% bootstrap CI, paired diffs, Holm correction, practical threshold (e.g. 0.01 PR-AUC).
- Pre-registration commit in W7 before final runs.

## Open decisions (lock in the phase noted; write the choice here)
1. **Unlabelled rows at retrain time (P2, critical).** A not-yet-released fraud must NOT be silently treated as legit. Proposal: fraud row label released at `t + delay_sample`; legit row counts as labelled legit only after maturity `t + d_max_setting`. Rows not yet matured are excluded from training and from "arrived-label" monitoring.
2. **Delay distribution (P2).** Fixed-mean shape, e.g. gamma/lognormal with mean = setting; 0-delay = exact. Seeded per replay.
3. **LOCKED 2026-11-18 — IEEE windows (t_days, half-open):** stream 0–182; held-out tail 154–182 (4 wk, 78,542 rows, 3.53% fraud). Phase-1 static split: train 0–112 (389,523; 3.43%), valid 112–133 (63,911; 4.11%) for early stopping/choices, test 133–154 (58,564; 3.26%) reported only. In `configs/base.yaml`; enforced by `fraud.splits.time_window` (raises PermissionError into held-out without FINAL_EVAL=1). Replay windows decided in P2 within 0–154.
4. **IEEE-CIS label propagation (P1).** `isFraud` is propagated to later txns of the same account, so card/uid aggregates can leak via label. Use only past features, never past *labels* in features unless released (label-aware features obey decision 1).
5. **Card/user id (P1).** No true id; build `uid` from `card1`+`addr1`+ (day − `D1`). Document it.
6. **CBPE needs calibrated probs (P2).** Class weights miscalibrate LightGBM → calibrate (isotonic) on a calibration slice of *released* labels. PR-AUC isn't in NannyML CBPE → estimate expected PR-AUC by Monte-Carlo sampling labels from calibrated probs (own code), also log CBPE ROC-AUC.
7. **Training window on retrain (P2):** sliding vs expanding — same for all policies.
8. **Retrain budget N, schedule period, drift thresholds (P2, before W7 commit)** — tuned only on the non-held-out part.
9. **LendingClub delay semantics (P1):** real default labels take months–years; we simulate the same delay grid and state this caveat. Time granularity is monthly → delays <30 d are effectively 0 or 1 month; state this.
10. **LOCKED 2026-11-18 — LC censoring cut-off.** File keeps only finished loans: quarterly volume halves at 2016Q2 (111.9k→62.9k) and default rate jumps 0.204→0.253, then falls to 0.024 by 2018Q4 → outcome-dependent selection, not drift. Stream = issue_d 2010-01 → 2015-12; held-out = 2015-07 → 2015-12. Residual risk: 60-month loans from ~2013+ may be partly censored too (no `term` column to check) → mention as a limitation.

## Phases
### Phase 1 — Data & leak-free baseline (W1–W4, 23 Nov – 20 Dec)
Build: repo skeleton, venv, CI (GitHub Actions: lint + pytest), Terraform skeleton, AWS Budgets $10 alert; download both datasets; pandera schemas; time-based split + window budget (decision 3); LR + LightGBM baselines in MLflow (local file store); point-in-time feature engineering (velocity, amount deviation, device/email); imbalance (class weights vs SMOTE-on-train-only), threshold tuning, cost metric.
Exit: datasets load + schema pass; CI green; budget alert received; LightGBM temporal ROC-AUC in published range (~0.90 ref) else fix pipeline; no-future-leak test passes; comparison table on validation window.

### Phase 2 — Research harness & results (W5–W9, 21 Dec – 24 Jan)
Build: replay simulator + label-delay service; scoring API (Lambda container) + Locust baseline (W6); 6 policies; drift injection; **pre-registration commit (W7)**; CBPE estimator; full grid (datasets × delays × policies × ≥10 replays) — on laptop or free Kaggle notebooks; stats script (bootstrap, paired, Holm); score held-out window **once**.
Exit: label-delay test (0 early labels); drift-detector check (fires on injected, quiet on control); reproducibility test; all runs in MLflow; one script → results table with CIs.

### Phase 3 — System, demo & write-up (W10–W12, 25 Jan – 14 Feb)
Build: k3s on t3.small (stop when idle) + ArgoCD GitOps; quality gate → promote → deploy; one SageMaker training job; monitoring dashboard; LLM case summaries for flagged txns; end-to-end demo; model card, README, demo video; CI mini-replication.
Exit: git push deploys a new model; SageMaker job logged; live demo (drift → retrain → deploy); clean checkout reproduces headline numbers.

## Accuracy rules (from plan, enforced)
1 pre-register W7 · 2 held-out tail scored once · 3 time splits only · 4 no-leak test · 5 reproduce baseline first · 6 ≥10 replays + bootstrap CI · 7 paired + Holm + practical threshold · 8 publish everything incl. losses.

## Stack
Python 3.12 · pandas/pyarrow · LightGBM · scikit-learn · PyTorch · pandera · river (ADWIN) · nannyml (CBPE ref) · MLflow · pytest · Locust · Docker · AWS Lambda/SageMaker/Terraform (ap-south-1) · k3s + ArgoCD · GitHub Actions.
Machine: RTX 4060 8 GB, Python 3.12.9, Docker 29, git 2.50; no terraform/aws CLI yet. Cost cap < $20; no EKS/NAT/LB/always-on endpoints.

## Status log
- 2026-11-18: Read plan PDF; created agent.md, context.md, CLAUDE.md. Blocked on Kaggle token for downloads.
- 2026-11-18: AWS CLI 2.41 + Terraform 1.17 installed (open a new shell / refresh PATH), region ap-south-1, budgets $1/$10/$50 exist. Renamed Data/→data/. Kaggle notebooks dropped (laptop enough). Plan PDF kept out of git (*.pdf ignored).
- 2026-11-18: Scaffolded repo (git init, no commits): pyproject, pinned requirements.txt (torch CPU), configs/base.yaml, src/fraud/{config,guards}.py (FINAL_EVAL held-out guard), 4 tests pass, ruff clean, .github/workflows/ci.yml. nannyml not installed (own CBPE planned; add later only as reference if deps allow).
- 2026-11-18: Pushed to private GitHub repo Nibiru7899/fraud-delay-retraining (2 commits). CI = startup_failure in 0s on both runs (YAML valid, Actions enabled) → account-level (billing/spending limit or private-repo minutes). Waiting on user.
- 2026-11-18: Repo made PUBLIC. CI now blocked by "account is locked due to a billing issue" (GitHub account-level; user to fix in Settings > Billing). Pinned CI runner to ubuntu-24.04 for reproducibility. Tests run locally meanwhile.
- 2026-11-18: Phase 1 started. data.py loaders + scripts/prepare_data.py → data/interim/{ieee,lc}.parquet + meta.json (hashes ieee 6a03affb78a5cde5, lc a03e54d2c4c83b46). pandera schemas (src/fraud/schemas.py) pass on both. Locked decisions 3 and 10. splits.py + tests (16 pass). MLflow → sqlite:///mlruns.db.
- 2026-11-18: Baselines (raw features, fixed params, seed 42), IEEE test 133–154: LightGBM ROC-AUC 0.9011 / PR-AUC 0.5336 / recall@1%FPR 0.4712; LR 0.8326 / 0.3608 / 0.2984. Valid (112–133) LGB PR-AUC 0.6759. **Rule 5 PASSED** (published temporal AUC 0.9004). Valid→test PR-AUC drop 0.14 over ~3 wk = evidence of decay (partly early-stopping optimism). Next: W3 point-in-time features (uid, velocity, amount deviation) + no-future-leak test.
- 2026-11-18: CI green on GitHub (billing fixed). Terraform skeleton (subagent): infra/terraform, S3 artifacts bucket (private, versioned, SSE), local state, validate OK, plan = 4 to add, NOT applied. CI job `terraform` (fmt/init/validate).
- 2026-11-18: W3 features v1 (src/fraud/features.py): card1/uid/P_emaildomain past-only counts & amount sums over 1/7/30d, past mean/std/z/ratio of amount, secs since last. Exact searchsorted + within-key cumsum → no-future-leak test passes with check_exact (handmade + 20k real rows at 2 cut-offs). First version used a global cumsum → float rounding differed under truncation; fixed. `*_n_prev` excluded (cumulative since file start = clock).
- 2026-11-18: v1 vs raw (fixed params): VALID PR-AUC 0.6759→0.6792; TEST 0.5336→0.5390, ROC 0.9011→0.9030.
- 2026-11-18: W4 imbalance (v1 feats, reports/imbalance_comparison.csv): VALID PR-AUC none 0.6792 / class_weight 0.6765 / SMOTENC(0.1) 0.6824 → diffs < 0.005 adoption bar → keep **none** (simpler, better-calibrated probs for CBPE). Cost metric (src/fraud/cost.py, exact threshold search): review cost $5/$10/$20 in config; thresholds chosen on VALID reach 97–99% of TEST-oracle net value at $5/$10 (none: $141.6k vs oracle $142.8k at $10; max catchable $288k).
- ITERATION RULE (locked): choices made on VALID PR-AUC only via scripts/eval_valid.py (test not computed); adopt a change only if VALID gain ≥ 0.005 or it removes a principled leak/clock risk. Test reported once at end of Phase 1; held-out tail untouched until Phase 2 end.
- 2026-11-18: v2 features (D-norm Dn = day−D, amt cents/log, uid past means of C/dist/Dn/V, past-only freq counts for 8 keys): VALID 0.6883; + drop raw D cols 0.6905 (adopted). v1 rerun reproduced 0.6792 exactly (real-data reproducibility ✓). Adversarial (train vs valid window) AUC 1.0; top: D1n, D3n, D5n, pemail_* aggregates, D10n, uid_secs_since_last, id_31 → ablating groups.
- 2026-11-18: Ablations + seeds (VALID): dropping Dn hurts (0.662); dropping pemail_* + card1/uid secs_since_last → 0.6991/0.6985/0.6992 (seeds 42/1/2) vs base 0.6905/0.6888/0.6959 → adopted (configs/base.yaml feature_set, reports/excl_final.txt, 482 feats). CatBoost GPU candidate (subagent, scripts/catboost_model.py): VALID 0.6551.
- 2026-11-18: Optuna (scripts/tune.py, 25 TPE trials, folds 0–91→91–112 and 0–112→112–133, lr 0.1): 0.6536→0.6721. Refit at lr 0.05: VALID 0.7127/0.7107 (seeds 42/1) vs 0.699 → adopted (reports/best_params.json). Caveat: fold B = VALID window, so VALID gain is optimistic; test untouched. optuna==5.1.0 pinned.
- 2026-11-18: **PHASE 1 DONE.** Final = LightGBM v2 feats (482) + tuned params (configs/lgb_tuned.json) + 3-seed avg; VALID 0.7142. CatBoost blend rejected (prob & rank blends both lower VALID). TEST (scored once): PR-AUC 0.5933 (baseline 0.5336), ROC 0.9247, rec@1%FPR 0.5283; net @$10 $160.6k (96% of test oracle). Model under-predicts on VALID (mean p 0.023 vs rate 0.041) → calibration matters for CBPE (open dec. 6). Report: docs/phase1_report.md. 32 tests pass, ruff clean. Next: Phase 2 (replay harness, delays, 6 policies).
