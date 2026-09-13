"""Time-aware validation.

The competition trains on days 1-35 and scores days 36-49, so every fold here
trains on a block of early days and is scored on the next 14 days. Shuffling
rows at random would let a model learn from the same day it is tested on,
which flatters it offline and then fails on the leaderboard.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HORIZON = 14  # days 36-49: the number of future days we ultimately predict


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: int  # train on every day up to and including this one
    val_start: int
    val_end: int

    @property
    def is_primary(self) -> bool:
        return self.val_end == 35


# Three expanding folds. The last one uses the most history and is the closest
# stand-in for the real task, so it is the number to steer by.
FOLDS = [
    Fold("days 16-29", train_end=15, val_start=16, val_end=29),
    Fold("days 19-32", train_end=18, val_start=19, val_end=32),
    Fold("days 22-35", train_end=21, val_start=22, val_end=35),
]

PRIMARY = FOLDS[-1]


def split(frame: pd.DataFrame, fold: Fold):
    """Boolean masks for one fold, based on the time_step column."""
    step = frame["time_step"]
    is_train = step <= fold.train_end
    is_val = step.between(fold.val_start, fold.val_end)
    return is_train.to_numpy(), is_val.to_numpy()


def per_step_auc(y_true, y_score, time_step) -> pd.DataFrame:
    """AUC for each day separately, plus how many illicit transactions it held."""
    frame = pd.DataFrame({"day": time_step, "y": y_true, "p": y_score})
    rows = []
    for day, part in frame.groupby("day"):
        both_classes = part["y"].nunique() == 2
        rows.append(
            {
                "day": int(day),
                "n": len(part),
                "illicit": int(part["y"].sum()),
                "auc": roc_auc_score(part["y"], part["p"]) if both_classes else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate(y_true, y_score, time_step, fold: Fold | None = None) -> dict:
    """Overall AUC plus the per-day breakdown for one validation window."""
    steps = per_step_auc(y_true, y_score, time_step)
    if fold is not None:
        steps["days_ahead"] = steps["day"] - fold.train_end
    return {
        "auc": roc_auc_score(y_true, y_score),
        "mean_daily_auc": steps["auc"].mean(),
        "worst_daily_auc": steps["auc"].min(),
        "per_step": steps,
    }


def print_report(name: str, result: dict) -> None:
    print(f"\n  {name}")
    print(
        f"    overall AUC {result['auc']:.4f}"
        f"   mean daily {result['mean_daily_auc']:.4f}"
        f"   worst day {result['worst_daily_auc']:.4f}"
    )
    steps = result["per_step"]
    ahead = steps.get("days_ahead")
    for _, row in steps.iterrows():
        marker = f" (+{int(row['days_ahead'])}d)" if ahead is not None else ""
        print(
            f"      day {int(row['day']):>2}{marker}  n={int(row['n']):>5}"
            f"  illicit={int(row['illicit']):>4}  AUC={row['auc']:.4f}"
        )


def decay_profile(results: dict[str, dict]) -> pd.DataFrame:
    """Average AUC by how many days ahead the prediction was, across folds.

    This is the honest picture of what happens on the leaderboard, where the
    test days sit 1 to 14 days past the end of training.
    """
    parts = [r["per_step"] for r in results.values() if "days_ahead" in r["per_step"]]
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts)
    return (
        combined.groupby("days_ahead")["auc"]
        .agg(["mean", "min", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_auc", "min": "worst_auc", "count": "folds"})
    )


def distribution_shift(train_frame, test_frame, columns, threshold=0.5):
    """Flag features that do not mean the same thing in train and test.

    RUN THIS ON EVERY ENGINEERED FEATURE BEFORE SUBMITTING.

    train.csv holds every node for days 1-35; test.csv holds only the labelled
    nodes for days 36-49, because the 46,668 unlabelled test-day transactions
    appear in the edge list but have no feature row. Any feature counting a
    transaction's neighbours therefore means something different on each side,
    and a fold-based check cannot see it: all folds live inside days 1-35 where
    the difference does not exist.

    A feature that is constant in training carries no signal the model can
    learn, and if it is non-constant at test time the model meets values it has
    never seen. That is worse than useless.

    Returns one row per feature, worst offenders first.
    """
    rows = []
    for column in columns:
        a = float(train_frame[column].mean())
        b = float(test_frame[column].mean())
        sd = float(train_frame[column].std())
        rows.append(
            {
                "feature": column,
                "train_mean": a,
                "test_mean": b,
                "train_sd": sd,
                # how far the test mean sits from the training distribution
                "shift_in_sds": abs(b - a) / sd if sd > 1e-9 else np.inf,
                "constant_in_train": sd < 1e-9,
            }
        )
    frame = pd.DataFrame(rows).sort_values("shift_in_sds", ascending=False)
    frame["suspect"] = frame["constant_in_train"] | (frame["shift_in_sds"] > threshold)
    return frame.reset_index(drop=True)


def assert_no_shift(train_frame, test_frame, columns, threshold=0.5):
    """Raise if any feature would reach a submission with a train/test mismatch."""
    report = distribution_shift(train_frame, test_frame, columns, threshold)
    bad = report[report["suspect"]]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} feature(s) differ between train and test:\n"
            + bad.head(10).to_string(index=False)
        )
    return report
