"""Local-only xgboost (feat_1-93, the transaction's own features) as a blend partner.

The mirror of model_neigh.py. Neighbour-only won pre-cliff offline and lost
0.008 on the leaderboard, as has every change the pre-cliff windows favoured.
Local-only is the reverse: pre-cliff its blend loses 0.003-0.008, post-cliff it
is neutral to +0.0005. Leaderboard test of the inversion pattern.

    python src/model_local.py 0.3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from data import LOCAL_FEATURES, SUBMISSIONS, load_labelled, load_test
from experiment import write_submission
from model_tuned import TUNED
from train import fit_final

BASE = "blend4_full40_d23_d18_d12.csv"  # best public, 0.95672


def main():
    weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    path = SUBMISSIONS / "local93_5seed.csv"
    if path.exists():
        neigh = pd.read_csv(path)["target"]
    else:
        neigh = pd.Series(fit_final(load_labelled(), load_test(), LOCAL_FEATURES, 79, TUNED))
        write_submission(neigh.to_numpy(), "local93_5seed")
    base = pd.read_csv(SUBMISSIONS / BASE)["target"].rank(pct=True)
    print(f"  spearman local93 vs best: {neigh.corr(base, method='spearman'):.3f}")
    blended = (1 - weight) * base + weight * neigh.rank(pct=True)
    name = f"blend4_local{int(weight * 100)}"
    print(f"    -> {write_submission(blended.to_numpy(), name).name}")


if __name__ == "__main__":
    main()
