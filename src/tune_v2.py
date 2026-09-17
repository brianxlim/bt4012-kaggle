"""Random search over settings combinations, on top of the 0.95333 model.

One-at-a-time sweeps have found nothing in this project twice now; every gain
has come from settings moving together. So this samples whole combinations
rather than walking one axis at a time.

Everything here is a *settings* change - the only category that has ever
transferred to the leaderboard (3 for 3, at about half the offline size).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from data import load_labelled, FEATURES
from train import cross_validate, summarise

BEST = dict(max_depth=10, colsample_bytree=0.5, learning_rate=0.1,
            scale_pos_weight=7.57, min_child_weight=8)
SPACE = dict(
    max_depth=[8, 10, 12, 14],
    colsample_bytree=[0.3, 0.4, 0.5, 0.7],
    subsample=[0.6, 0.7, 0.8, 1.0],
    min_child_weight=[3, 5, 8, 12, 20],
    reg_lambda=[1.0, 3.0, 10.0, 30.0],
    reg_alpha=[0.0, 0.5, 2.0],
    gamma=[0.0, 0.2, 1.0],
)
N = 24
rng = np.random.default_rng(4012)
lab = load_labelled()

runs, rounds = cross_validate(lab, FEATURES, BEST)
base = summarise(runs)
base_pf = [base["per_fold"][f] for f in ["days 16-29", "days 19-32", "days 22-35"]]
print(f"{'#':>3}  {'AUC':>7} {'se':>7} {'delta':>8} {'folds':>6}  config")
print(f"{'cur':>3}  {base['auc']:>7.4f} {base['stderr']:>7.4f} {'':>8} {'':>6}  (current best)", flush=True)

rows = []
for i in range(N):
    cfg = {**BEST, **{k: rng.choice(v).item() for k, v in SPACE.items()}}
    r, rd = cross_validate(lab, FEATURES, cfg)
    s = summarise(r)
    pf = [s["per_fold"][f] for f in ["days 16-29", "days 19-32", "days 22-35"]]
    wins = sum(p > b for p, b in zip(pf, base_pf))
    short = " ".join(f"{k.replace('colsample_bytree','cs').replace('min_child_weight','mcw')
        .replace('max_depth','d').replace('subsample','sub').replace('reg_lambda','L2')
        .replace('reg_alpha','L1')}={v}" for k, v in cfg.items()
        if k not in ("learning_rate", "scale_pos_weight"))
    print(f"{i:>3}  {s['auc']:>7.4f} {s['stderr']:>7.4f} {s['auc']-base['auc']:>+8.4f} {wins:>4}/3  {short}", flush=True)
    rows.append({**cfg, "auc": s["auc"], "stderr": s["stderr"], "delta": s["auc"]-base["auc"],
                 "folds_up": wins, "rounds": rd})

f = pd.DataFrame(rows).sort_values("delta", ascending=False)
f.to_csv("reports/tuning_v2.csv", index=False)
print("\n=== top 5 ===")
print(f.head(5)[["max_depth","colsample_bytree","subsample","min_child_weight",
                "reg_lambda","reg_alpha","gamma","auc","stderr","delta","folds_up"]].to_string(index=False))
print(f"\ncandidates beating current by > 0.005 and up in all 3 folds: "
      f"{int(((f.delta > 0.005) & (f.folds_up == 3)).sum())}")
