"""Other model families fit on every labelled day, submitted on their own.

Offline windows have misjudged every model family (LightGBM and ExtraTrees won
10/10 post-cliff runs and lost on the board), so the leaderboard is the only
judge. Each is fit on all 165 features, 5 seeds rank-averaged.

    python src/model_alt.py catboost
    python src/model_alt.py rf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from data import FEATURES, load_labelled, load_test
from experiment import write_submission

MODELS = {
    # symmetric (oblivious) trees and ordered boosting: built to resist overfitting
    "catboost": lambda s: CatBoostClassifier(
        iterations=600, learning_rate=0.05, depth=6, auto_class_weights="Balanced",
        random_seed=s, verbose=0, thread_count=4),
    # the model the original Elliptic paper found best
    "rf": lambda s: RandomForestClassifier(
        n_estimators=1000, max_features="sqrt", min_samples_leaf=1, n_jobs=4, random_state=s),
}


def main():
    name = sys.argv[1]
    labelled, test = load_labelled(), load_test()
    X, y, Xt = labelled[FEATURES].to_numpy(), labelled["label"].to_numpy(), test[FEATURES].to_numpy()
    scores = np.zeros(len(test))
    for seed in range(5):
        p = MODELS[name](seed).fit(X, y).predict_proba(Xt)[:, 1]
        scores += pd.Series(p).rank(pct=True).to_numpy()
    path = write_submission(scores / 5, f"{name}_5seed")
    best = pd.read_csv(path.parent / "blend4_full40_d23_d18_d12.csv")["target"]
    print(f"  {name}: spearman vs best {pd.Series(scores).corr(best, method='spearman'):.3f} -> {path.name}")


if __name__ == "__main__":
    main()
