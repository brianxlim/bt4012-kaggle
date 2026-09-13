"""Shared plumbing: writing submissions and keeping a record of what scored what."""

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from data import REPORTS, SUBMISSIONS, load_test

LOG = REPORTS / "experiments.csv"

# Fixed column order. Every run writes the same columns so the file stays
# readable as a table and rows from different experiments line up.
COLUMNS = [
    "run_at",
    "name",
    "auc",
    "stderr",
    "worst_day",
    "n_features",
    "rounds",
    "lb_public",
    "notes",
]


def write_submission(scores, name: str) -> Path:
    """Write index,target in the exact row order the competition expects."""
    test = load_test()
    if len(scores) != len(test):
        raise ValueError(f"expected {len(test)} scores, got {len(scores)}")
    frame = pd.DataFrame({"index": test["index"].to_numpy(), "target": scores})
    SUBMISSIONS.mkdir(exist_ok=True)
    path = SUBMISSIONS / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def log_experiment(name, summary, n_features, rounds, notes=None) -> None:
    """Append one row per run so model-to-model comparisons survive the session."""
    row = {
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "auc": round(summary["auc"], 5),
        "stderr": round(summary["stderr"], 5),
        "worst_day": round(summary["worst_day"], 5),
        "n_features": n_features,
        "rounds": rounds,
        "lb_public": "",
        "notes": json.dumps(notes or {}),
    }
    REPORTS.mkdir(exist_ok=True)
    frame = pd.DataFrame([row], columns=COLUMNS)
    frame.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


def record_leaderboard(name: str, score: float) -> None:
    """Fill in what the leaderboard actually said once a submission is scored."""
    log = pd.read_csv(LOG)
    log.loc[log["name"] == name, "lb_public"] = score
    log.to_csv(LOG, index=False)
