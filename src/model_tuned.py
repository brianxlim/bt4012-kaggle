"""The model of record. Leaderboard 0.95333.

Tuned xgboost on the raw 165 features, labelled rows only.

The two class-balance settings are the reason this beats the plain tuned model.
Only 11.67% of labelled rows are illicit, so `scale_pos_weight` counts each one
as worth 7.57 licit rows — the ratio that makes both groups weigh the same.
That alone makes tiny illicit clusters look more important than they are, so
`min_child_weight` raises the evidence a group must carry before a tree may
split it, which stops the model carving those clusters out.

Neither works alone: min_child_weight on its own measured -0.0056 and
scale_pos_weight on its own +0.0046. Together, +0.0090 offline over 3 folds x
10 seeds, and +0.0052 on the leaderboard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data import FEATURES, load_labelled, load_test
from train import run

TUNED = dict(
    max_depth=10,
    colsample_bytree=0.5,
    learning_rate=0.1,
    scale_pos_weight=7.57,   # 27,591 licit / 3,644 illicit
    min_child_weight=8,
)


def main():
    labelled, test = load_labelled(), load_test()
    run("spw757_mcw8", labelled, test, FEATURES, params=TUNED)


if __name__ == "__main__":
    main()
