"""Neighbour-only xgboost (feat_94-165) as a blend partner.

Offline, across the regime cliff at day 23 the neighbour summaries hold up and
the transaction's own features do not: trained on days <=15 and scored on
16-29, neighbour-only 0.9453 vs all features 0.9163. Within one regime it is the
reverse (0.9299 vs 0.9972 on days 30-35). The two rank rows differently
(Spearman 0.43-0.62). A 20% share gains +0.008 to +0.017 pre-cliff and loses
0.001 to 0.005 post-cliff. Days 36-49 include the day-43 break, so this is a
leaderboard test of which regime the test period behaves like.

    python src/model_neigh.py 0.2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from data import NEIGHBOUR_FEATURES, SUBMISSIONS, load_labelled, load_test
from experiment import write_submission
from model_tuned import TUNED
from train import fit_final

BASE = "blend4_full40_d23_d18_d12.csv"  # best public, 0.95672


def main():
    weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    path = SUBMISSIONS / "neigh72_5seed.csv"
    if path.exists():
        neigh = pd.read_csv(path)["target"]
    else:
        neigh = pd.Series(fit_final(load_labelled(), load_test(), NEIGHBOUR_FEATURES, 79, TUNED))
        write_submission(neigh.to_numpy(), "neigh72_5seed")
    base = pd.read_csv(SUBMISSIONS / BASE)["target"].rank(pct=True)
    print(f"  spearman neigh72 vs best: {neigh.corr(base, method='spearman'):.3f}")
    blended = (1 - weight) * base + weight * neigh.rank(pct=True)
    name = f"blend4_neigh{int(weight * 100)}"
    print(f"    -> {write_submission(blended.to_numpy(), name).name}")


if __name__ == "__main__":
    main()
