"""Post-processing: mix each score with its rank inside its own day.

The model scores days 43-49 almost wholly clean (under 1% of rows above 0.5,
against 6-14% on days 37-42), in line with the known change in illicit activity
around day 43. If illicit transactions there are ones the model cannot
recognise, they sit below many clean day-36-42 rows, and pooled AUC pays for
every such pair. Mixing in the within-day rank lets each day's most suspicious
rows compete.

Offline it only loses (days 1-35, where the model's day-to-day balance is
right): -0.003 at 25%, -0.009 at 50%. The test period is where it could differ,
so this is a leaderboard experiment.

    python src/dayrank.py base.csv 0.25 out.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data import SUBMISSIONS, load_test


def main():
    base, share, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    frame = pd.read_csv(SUBMISSIONS / base)
    day = load_test()["time_step"].to_numpy()
    overall = frame["target"].rank(pct=True)
    within = frame.assign(day=day).groupby("day")["target"].rank(pct=True)
    frame["target"] = (1 - share) * overall + share * within
    frame.to_csv(SUBMISSIONS / out, index=False)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
