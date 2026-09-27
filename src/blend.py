"""Rank-average submitted prediction files: python src/blend.py out.csv a.csv:w b.csv:w ..."""

import sys
from pathlib import Path

import pandas as pd

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"


def main():
    out, parts = sys.argv[1], sys.argv[2:]
    total, blended = 0.0, None
    for part in parts:
        name, weight = part.rsplit(":", 1)
        frame = pd.read_csv(SUBMISSIONS / name)
        ranks = frame["target"].rank(pct=True) * float(weight)
        blended = ranks if blended is None else blended + ranks
        total += float(weight)
    pd.DataFrame({"index": frame["index"], "target": blended / total}).to_csv(SUBMISSIONS / out, index=False)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
