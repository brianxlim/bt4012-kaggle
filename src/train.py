"""One training routine shared by every experiment, so results stay comparable.

Every score here is averaged over several random seeds. A single run of this
model swings by about 0.013 AUC purely from its own randomness, which is wider
than the entire gap between first and last place on the leaderboard. Judging a
feature idea on one run therefore tells you almost nothing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import validation as v
from experiment import log_experiment, write_submission

PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)

SEEDS = (0, 1, 2, 3, 4)


def cross_validate(train_frame, columns, params=None, seeds=SEEDS):
    """Fit every fold with every seed. Returns per-run scores and best tree counts."""
    params = {**PARAMS, **(params or {})}
    X = train_frame[columns].to_numpy(dtype="float32")
    y = train_frame["label"].to_numpy()
    days = train_frame["time_step"].to_numpy()

    runs, rounds = [], []
    for fold in v.FOLDS:
        is_train, is_val = v.split(train_frame, fold)
        for seed in seeds:
            model = XGBClassifier(
                **{**params, "random_state": seed}, early_stopping_rounds=100
            )
            model.fit(
                X[is_train], y[is_train], eval_set=[(X[is_val], y[is_val])], verbose=False
            )
            scores = model.predict_proba(X[is_val])[:, 1]
            per_day = v.per_step_auc(y[is_val], scores, days[is_val])
            runs.append(
                {
                    "fold": fold.name,
                    "seed": seed,
                    "auc": roc_auc_score(y[is_val], scores),
                    "worst_day": per_day["auc"].min(),
                }
            )
            if fold.is_primary:
                rounds.append(model.best_iteration + 1)
    return pd.DataFrame(runs), int(np.mean(rounds))


def summarise(runs: pd.DataFrame) -> dict:
    """Headline number is the mean across every fold and seed, with its error bar."""
    return {
        "auc": runs["auc"].mean(),
        "stderr": runs["auc"].std() / np.sqrt(len(runs)),
        "worst_day": runs["worst_day"].mean(),
        "per_fold": runs.groupby("fold")["auc"].mean().to_dict(),
    }


def report(name: str, summary: dict) -> None:
    folds = "  ".join(f"{k.split()[-1]} {v:.4f}" for k, v in summary["per_fold"].items())
    print(
        f"  {name:30s} {summary['auc']:.4f} +/- {summary['stderr']:.4f}"
        f"   worst day {summary['worst_day']:.4f}   [{folds}]"
    )


def fit_final(train_frame, test_frame, columns, rounds, params=None, seeds=SEEDS):
    """Average the predictions of several seeds.

    Averaging cancels some of each run's private randomness, so it is a small
    free gain on the leaderboard rather than just a tidier measurement.
    """
    params = {**PARAMS, **(params or {}), "n_estimators": rounds}
    X = train_frame[columns].to_numpy(dtype="float32")
    y = train_frame["label"].to_numpy()
    X_test = test_frame[columns].to_numpy(dtype="float32")

    scores = np.zeros(len(test_frame))
    for seed in seeds:
        model = XGBClassifier(**{**params, "random_state": seed})
        model.fit(X, y, verbose=False)
        # Rank before averaging: only the ordering is scored, and ranks stop one
        # seed's extreme probabilities from dominating the blend.
        scores += pd.Series(model.predict_proba(X_test)[:, 1]).rank(pct=True).to_numpy()
    return scores / len(seeds)


def run(name, train_frame, test_frame, columns, params=None, seeds=SEEDS, submit=True):
    runs, rounds = cross_validate(train_frame, columns, params, seeds)
    summary = summarise(runs)
    report(name, summary)
    if not submit:
        return summary, None

    # More rows in the final fit than in a fold, so it needs proportionally more trees.
    is_train, _ = v.split(train_frame, v.PRIMARY)
    n_rounds = int(rounds * len(train_frame) / is_train.sum())
    scores = fit_final(train_frame, test_frame, columns, n_rounds, params, seeds)
    path = write_submission(scores, name)
    log_experiment(
        name,
        summary,
        n_features=len(columns),
        rounds=n_rounds,
        notes={
            "seeds": list(seeds),
            "per_fold": {k: round(x, 4) for k, x in summary["per_fold"].items()},
        },
    )
    print(f"    -> {path.name}  ({len(columns)} features, {n_rounds} trees x {len(seeds)} seeds)")
    return summary, scores


def importances(train_frame, columns, params=None, top=20):
    params = {**PARAMS, **(params or {}), "n_estimators": 300}
    model = XGBClassifier(**params)
    model.fit(train_frame[columns].to_numpy(dtype="float32"), train_frame["label"].to_numpy())
    return (
        pd.Series(model.feature_importances_, index=columns)
        .sort_values(ascending=False)
        .head(top)
    )
