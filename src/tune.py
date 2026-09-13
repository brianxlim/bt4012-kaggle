"""Search XGBoost settings. The defaults were never checked against this data.

Each candidate is scored the same way as everything else: averaged over folds
and seeds, with an error bar. Seeds are trimmed to 3 here because a search runs
many candidates; the winner is re-checked on the full 5 afterwards.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import itertools

import pandas as pd

from data import FEATURES, load_labelled
from train import cross_validate, summarise

SEARCH_SEEDS = (0, 1, 2)

GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 20],
    "subsample": [0.6, 0.8],
    "colsample_bytree": [0.4, 0.8],
}


def one_at_a_time(labelled, base=None):
    """Vary one setting at a time from the current defaults.

    A full grid would be 108 candidates and hours of fitting. Varying one
    setting at a time costs 13 and shows which settings matter at all, which is
    what we actually need before spending time on a finer search.
    """
    base = base or {}
    rows = []
    runs, _ = cross_validate(labelled, FEATURES, base, SEARCH_SEEDS)
    reference = summarise(runs)
    rows.append({"setting": "current defaults", "value": "-", **_row(reference)})
    print(f"  {'current defaults':34s} {reference['auc']:.4f} +/- {reference['stderr']:.4f}")

    for setting, options in GRID.items():
        for value in options:
            if base.get(setting) == value:
                continue
            params = {**base, setting: value}
            runs, _ = cross_validate(labelled, FEATURES, params, SEARCH_SEEDS)
            summary = summarise(runs)
            delta = summary["auc"] - reference["auc"]
            mark = "  <-" if delta > 2 * summary["stderr"] else ""
            print(
                f"  {setting + '=' + str(value):34s} {summary['auc']:.4f}"
                f" +/- {summary['stderr']:.4f}   {delta:+.4f}{mark}"
            )
            rows.append({"setting": setting, "value": value, **_row(summary)})
    return pd.DataFrame(rows)


def _row(summary):
    return {
        "auc": summary["auc"],
        "stderr": summary["stderr"],
        "worst_day": summary["worst_day"],
    }


if __name__ == "__main__":
    labelled = load_labelled()
    print(f"\n  3 folds x {len(SEARCH_SEEDS)} seeds per candidate\n")
    table = one_at_a_time(labelled)
    table.to_csv(Path(__file__).parents[1] / "reports" / "tuning.csv", index=False)
    print("\n  best:")
    print(table.nlargest(5, "auc").to_string(index=False))
