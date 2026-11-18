"""MLflow setup + provenance tags (seed, data hash, git commit) on every run."""
from __future__ import annotations

import json
import subprocess

import mlflow

from fraud.config import ROOT, load_config


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return sha + ("-dirty" if dirty.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def start_run(run_name: str, dataset: str, seed: int):
    cfg = load_config()
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + (ROOT / uri.removeprefix("sqlite:///")).as_posix()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    meta = json.loads((ROOT / "data/interim/meta.json").read_text())
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tags(
        {"git_commit": _git_commit(), "dataset": dataset, "data_hash": meta[dataset]["hash"]}
    )
    mlflow.log_param("seed", seed)
    return run
