"""Baseline: XGBoost on the 165 supplied features, labelled rows only.

Deliberately plain. Its job is to give an offline number that can be compared
against the leaderboard, so that later offline numbers can be trusted without
spending one of the three daily submissions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from xgboost import XGBClassifier

import validation as v
from data import FEATURES, load_labelled, load_test
from experiment import log_experiment, write_submission

PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    tree_method="hist",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)


def main() -> None:
    labelled = load_labelled()
    X = labelled[FEATURES].to_numpy()
    y = labelled["label"].to_numpy()
    days = labelled["time_step"].to_numpy()

    # time_step is not a model input: days 36-49 never appear in training, so a
    # tree can only lump them in with day 35. It is used to split, nothing more.
    results, best_rounds = {}, {}
    for fold in v.FOLDS:
        is_train, is_val = v.split(labelled, fold)
        model = XGBClassifier(**PARAMS, early_stopping_rounds=100)
        model.fit(
            X[is_train],
            y[is_train],
            eval_set=[(X[is_val], y[is_val])],
            verbose=False,
        )
        scores = model.predict_proba(X[is_val])[:, 1]
        results[fold.name] = v.evaluate(y[is_val], scores, days[is_val], fold)
        best_rounds[fold.name] = model.best_iteration + 1
        v.print_report(f"{fold.name}  (trained on days 1-{fold.train_end})", results[fold.name])

    print("\n  How fast the model goes stale")
    profile = v.decay_profile(results)
    for _, row in profile.iterrows():
        print(
            f"    {int(row['days_ahead']):>2} days ahead:"
            f"  mean AUC {row['mean_auc']:.4f}   worst {row['worst_auc']:.4f}"
        )

    primary = results[v.PRIMARY.name]
    print(f"\n  Headline (days 22-35 fold): AUC {primary['auc']:.4f}")

    # Refit on every labelled day, scaling the tree count up in proportion to
    # the extra rows the final model gets to see.
    is_train, _ = v.split(labelled, v.PRIMARY)
    rounds = int(best_rounds[v.PRIMARY.name] * len(y) / is_train.sum())
    final = XGBClassifier(**{**PARAMS, "n_estimators": rounds})
    final.fit(X, y, verbose=False)

    test = load_test()
    scores = final.predict_proba(test[FEATURES].to_numpy())[:, 1]
    path = write_submission(scores, "baseline_xgb")
    log_experiment(
        "baseline_xgb",
        results,
        {"rounds": rounds, "fold_rounds": best_rounds, "features": "raw 165"},
    )

    print(f"\n  Final model: {rounds} trees on all {len(y):,} labelled rows")
    print(f"  Submission:  {path}")
    print(f"  Predicted illicit rate (score > 0.5): {(scores > 0.5).mean():.3%}")


if __name__ == "__main__":
    main()
