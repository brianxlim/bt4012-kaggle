"""Shift the pooled rank of every test row from a given day on: python src/blockshift.py in.csv out.csv 43 -0.03

Days 43-49 come after a dark-market shutdown. Pooled AUC compares rows across days, so a
constant offset on that block reorders it against days 36-42 without touching the order
inside either block.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from data import SUBMISSIONS, load_test


def main():
    source, out, first_day, delta = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    frame = pd.read_csv(SUBMISSIONS / source)
    rank = frame["target"].rank(pct=True)
    block = load_test()["time_step"].to_numpy() >= first_day
    rank[block] += delta
    pd.DataFrame({"index": frame["index"], "target": rank}).to_csv(SUBMISSIONS / out, index=False)
    print(f"  {block.sum()} rows from day {first_day} shifted by {delta:+} -> {out}")


if __name__ == "__main__":
    main()
