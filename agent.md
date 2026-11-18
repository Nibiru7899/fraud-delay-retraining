# agent.md — operating rules for AI agents on this repo

Auto-loaded via `CLAUDE.md`. Keep this file short; details live in `context.md`.

## Project in one line
Measure **when to retrain** a LightGBM fraud model when labels arrive 0/7/30/60 days late,
comparing 6 policies (incl. label-free CBPE trigger and a hybrid), on real data only, with
pre-registered stats. Plus a thin deployed system (Lambda API, MLflow, k3s+ArgoCD).

## Token-saving rules (follow every session)
1. Read `context.md` **only the section you need** (use Grep on the `## ` headings). Do not re-read the PDF; `context.md` is its distilled version.
2. Check `## Status log` in `context.md` first to see where we are. Append one line per finished step; never rewrite history.
3. Never print raw DataFrames or full logs. Scripts print **≤20-line summaries** (shape, nulls, key metrics). Full outputs go to `reports/` or MLflow.
4. Data files never get `Read`. Inspect with a short pandas one-liner (`.shape`, `.dtypes.value_counts()`, `.head(3)`).
5. Long jobs (downloads, replays, training) run in background; don't poll.
6. Broad code search → `Explore` subagent; otherwise Grep/Glob directly.
7. Don't re-derive decisions in `## Locked decisions`. Change one only if the user agrees, and log why.

## Correctness rules (non-negotiable)
- **No synthetic data**, except the explicitly labelled *injected-drift* variant of IEEE-CIS, always reported separately.
- **Time-ordered splits only.** No random splits, no shuffling across time.
- **Point-in-time features:** any per-card/user aggregate uses only rows with time `< t` (strict). Covered by the no-future-leak test.
- **Label visibility:** a model/monitor at time `t` may only use labels with `release_time <= t`. Covered by the label-delay test.
- **Held-out final window** is never touched until the single final scoring (Phase 2 end). Code must refuse to load it unless `FINAL_EVAL=1`.
- Fixed seeds everywhere; record seed, data hash and git commit in every MLflow run.
- Same replay stream, delays and retrain budget for every policy (paired design).
- Report losses and ties honestly. Never tune on test windows.

## Conventions
- Python 3.12, venv at `.venv`, deps pinned in `requirements.txt`.
- Layout: `src/fraud/` (package), `scripts/` (entry points), `tests/`, `configs/` (YAML), `data/raw|interim|processed` (git-ignored), `reports/`, `infra/` (Terraform, k8s).
- Datasets live in `data/` (user-provided, git-ignored). Paths are in `context.md` → Datasets. New downloads go in their own empty dir under `data/raw/<name>/`; read data with `python -I`.
- Run Python via `.venv/Scripts/python` (Windows venv).
- Windows host: Bash tool = Git Bash; use forward slashes.
- Commit only when the user asks. Never commit data or secrets.
