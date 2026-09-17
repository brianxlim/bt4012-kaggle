"""The model of record. Leaderboard 0.95521.

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

The tree-shape settings on top of that came from a 24-configuration random
search over whole combinations (src/tune_v2.py). Every configuration that beat
the previous model shared subsample=1.0 and colsample_bytree=0.7, so the ridge
was tested directly rather than taking the single best sample: +0.0110 on seeds
0-9 and +0.0115 on seeds 10-19, which the search never saw.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data import FEATURES, load_labelled, load_test
from train import run

TUNED = dict(
    max_depth=12,
    colsample_bytree=0.7,
    subsample=1.0,
    learning_rate=0.1,
    scale_pos_weight=7.57,   # 27,591 licit / 3,644 illicit
    min_child_weight=8,
    gamma=0.2,               # a split must earn this much before it is taken
)


def main():
    labelled, test = load_labelled(), load_test()
    run("ridge_d12cs07sub10g02", labelled, test, FEATURES, params=TUNED)


if __name__ == "__main__":
    main()
