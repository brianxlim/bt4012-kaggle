"""LightGBM and ExtraTrees as blend partners, judged on post-cliff windows.

The test days behave like the post-cliff regime (a neighbour-only partner that
won pre-cliff lost 0.008 on the board). So partners are judged on five
post-cliff windows (train <=k, score the next 6 days, k = 23..31, 2 seeds):

    partner              alone vs xgb   30% blend   50% blend   wins
    LightGBM             +0.0001        +0.0004     +0.0005     10/10
    ExtraTrees           -0.0020        +0.0010     +0.0009     10/10, 9/10
    RandomForest         -0.0062        -0.0002     -0.0012

Earlier these families were rejected on the pre-cliff windows, which have
misjudged every non-settings change. Each partner is fit on all labelled days,
5 seeds, and rank-blended into the best public file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier

from data import FEATURES, SUBMISSIONS, load_labelled, load_test
from experiment import write_submission

BASE = "blend4_full40_d23_d18_d12.csv"  # best public, 0.95672
SEEDS = range(5)

MODELS = {
    "lgbm": lambda s: LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63, colsample_bytree=0.7,
        min_child_samples=10, scale_pos_weight=7.57, random_state=s, verbose=-1, n_jobs=4),
    "et": lambda s: ExtraTreesClassifier(
        500, max_features=0.5, min_samples_leaf=1, n_jobs=4, random_state=s),
}


def fit(name):
    path = SUBMISSIONS / f"{name}_5seed.csv"
    if path.exists():
        return pd.read_csv(path)["target"]
    labelled, test = load_labelled(), load_test()
    X, y, Xt = labelled[FEATURES].to_numpy(), labelled["label"].to_numpy(), test[FEATURES].to_numpy()
    scores = np.zeros(len(test))
    for seed in SEEDS:
        p = MODELS[name](seed).fit(X, y).predict_proba(Xt)[:, 1]
        scores += pd.Series(p).rank(pct=True).to_numpy()
    write_submission(scores / len(SEEDS), f"{name}_5seed")
    return pd.Series(scores)


def main():
    base = pd.read_csv(SUBMISSIONS / BASE)["target"].rank(pct=True)
    parts = {name: fit(name).rank(pct=True) for name in MODELS}
    for name, p in parts.items():
        print(f"  spearman {name} vs best: {p.corr(base, method='spearman'):.3f}")
    write_submission((0.7 * base + 0.3 * parts["et"]).to_numpy(), "blend4_et30")
    write_submission((0.5 * base + 0.5 * parts["lgbm"]).to_numpy(), "blend4_lgbm50")
    write_submission((0.5 * base + 0.25 * parts["lgbm"] + 0.25 * parts["et"]).to_numpy(), "blend4_lgbm25_et25")
    print("    -> blend4_et30.csv, blend4_lgbm50.csv, blend4_lgbm25_et25.csv")


if __name__ == "__main__":
    main()
