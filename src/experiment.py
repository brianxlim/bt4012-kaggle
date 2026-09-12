"""Shared plumbing: writing submissions and keeping a record of what scored what."""

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from data import REPORTS, SUBMISSIONS, load_test

LOG = REPORTS / "experiments.csv"


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


def log_experiment(name: str, results: dict, extra: dict | None = None) -> None:
    """Append one row per run so model-to-model comparisons survive the session."""
    row = {
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "name": name,
        **{f"auc_{fold.replace(' ', '_')}": r["auc"] for fold, r in results.items()},
        **{
            f"worst_day_{fold.replace(' ', '_')}": r["worst_daily_auc"]
            for fold, r in results.items()
        },
    }
    if extra:
        row["notes"] = json.dumps(extra)
    REPORTS.mkdir(exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
