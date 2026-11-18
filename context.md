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
3. **IEEE-CIS stream length vs 60-day delay (P1).** ~182 days total: initial train window + held-out final window + 60-day delay leaves little room. Must compute the window budget in P1 and maybe shorten held-out to ~3–4 weeks. Write the numbers here.
4. **IEEE-CIS label propagation (P1).** `isFraud` is propagated to later txns of the same account, so card/uid aggregates can leak via label. Use only past features, never past *labels* in features unless released (label-aware features obey decision 1).
5. **Card/user id (P1).** No true id; build `uid` from `card1`+`addr1`+ (day − `D1`). Document it.
6. **CBPE needs calibrated probs (P2).** Class weights miscalibrate LightGBM → calibrate (isotonic) on a calibration slice of *released* labels. PR-AUC isn't in NannyML CBPE → estimate expected PR-AUC by Monte-Carlo sampling labels from calibrated probs (own code), also log CBPE ROC-AUC.
7. **Training window on retrain (P2):** sliding vs expanding — same for all policies.
8. **Retrain budget N, schedule period, drift thresholds (P2, before W7 commit)** — tuned only on the non-held-out part.
9. **LendingClub delay semantics (P1):** real default labels take months–years; we simulate the same delay grid and state this caveat.

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
