"""The tuned model: same 165 features, better settings.

The search (tune.py, then combination runs) landed on deeper trees, a faster
learning rate, and each tree seeing about half the columns. Every setting that
made the model more cautious hurt, some badly, which says this problem needs
specific rules rather than broad ones - fraud is a small minority with
distinctive patterns, and a cautious model averages those patterns away.

Several settings around this point score the same within their error bars. This
one is chosen because it has the tightest error bar of the tied group, not
because its average is highest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data import FEATURES, load_labelled, load_test
from train import run

TUNED = dict(max_depth=10, colsample_bytree=0.5, learning_rate=0.1)


def main() -> None:
    labelled = load_labelled()
    test = load_test()
    print("\n  3 folds x 5 seeds = 15 runs\n")
    run("tuned_d10cs05lr01", labelled, test, FEATURES, params=TUNED)


if __name__ == "__main__":
    main()
