"""Model of record plus one feature fix: drifting features become within-day ranks.

Some supplied features sit in a completely different range on days 36-49 than
on days 1-35 (feat_101: over 99% of test rows are above the training maximum).
A tree clamps every such value into its top bin, so on test the feature stops
distinguishing anything. Dropping those features (stable144) threw away real
signal and lost 0.0083 on the leaderboard.

Here each drifting feature is instead replaced by its percentile rank among the
labelled transactions of the same day. The rank keeps the ordering inside a day,
which is where the fraud signal lives, and removes the level, which is what
drifts. It is computed the same way on both sides: over labelled rows per day in
train, over test.csv rows per day in test (test.csv holds only labelled rows).

"Drifting" means the feature alone separates a training row from a test row at
AUC above 0.75. That picks 17 features. The choice sits on a plateau: every
threshold from 0.65 to 0.80 gained, only 0.90 (6 features) did not.

Offline, paired against the model of record over the pre-cliff folds:
+0.0085 on seeds 5-14, +0.0118 on seeds 15-24 (27/30 wins). On the saturated
post-cliff windows +0.0004 with 9/10 wins, so it is not paying for the
pre-cliff gain with post-cliff loss.

Settings and tree count are unchanged from model_tuned.py, so only the
features differ.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data import FEATURES, load_labelled, load_test
from experiment import log_experiment, write_submission
from model_tuned import TUNED
from train import cross_validate, fit_final, report, summarise

THRESHOLD = 0.75
ROUNDS = 79  # the model of record's tree count


def drifting(train_frame, test_frame, threshold=THRESHOLD):
    """Features that alone tell a training row from a test row above `threshold` AUC."""
    is_test = np.r_[np.zeros(len(train_frame)), np.ones(len(test_frame))]
    out = []
    for column in FEATURES:
        auc = roc_auc_score(is_test, np.r_[train_frame[column], test_frame[column]])
        if max(auc, 1 - auc) > threshold:
            out.append(column)
    return out


def with_day_ranks(frame, columns):
    """Add r_<feature>: the feature's percentile rank within its own day."""
    ranks = frame.groupby("time_step")[columns].rank(pct=True).astype("float32")
    ranks.columns = [f"r_{c}" for c in columns]
    return pd.concat([frame, ranks], axis=1)


def main():
    labelled, test = load_labelled(), load_test()
    drift = drifting(labelled, test)
    print(f"  {len(drift)} drifting features: {', '.join(drift)}")
    labelled, test = with_day_ranks(labelled, drift), with_day_ranks(test, drift)
    columns = [c for c in FEATURES if c not in drift] + [f"r_{c}" for c in drift]

    name = "driftrank75"
    params = {**TUNED, "n_estimators": ROUNDS}
    runs, _ = cross_validate(labelled, columns, params)
    summary = summarise(runs)
    report(name, summary)

    scores = fit_final(labelled, test, columns, ROUNDS, TUNED)
    path = write_submission(scores, name)
    log_experiment(
        name, summary, n_features=len(columns), rounds=ROUNDS,
        notes={"drift_threshold": THRESHOLD, "drifting": drift},
    )
    print(f"    -> {path.name}  ({len(columns)} features, {ROUNDS} trees x 5 seeds)")


if __name__ == "__main__":
    main()
