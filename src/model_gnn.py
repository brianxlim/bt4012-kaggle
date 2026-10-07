"""Final GNN fit on every labelled day, 5 seeds rank-averaged, and its blend.

Offline (3 seeds, 250 epochs): alone it trails xgboost (0.9928 vs 0.9972 on
days 30-35). Blended at 20% it gains +0.0005 to +0.0008 on the post-cliff
windows and loses 0.004 on the pre-cliff one. It ranks rows differently from
xgboost (Spearman 0.72-0.76), which is what a blend partner needs.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from data import SUBMISSIONS, load_test
from experiment import write_submission
from gnn import GraphData, fit_predict

EPOCHS = 250
SEEDS = range(5)
WEIGHT = 0.2
BASE = "blend3_full50_d23_d18.csv"  # best public blend, 0.95646


def main():
    data = GraphData()
    test = load_test()
    rows = data.frame.index.get_indexer(test["txId"])
    train_mask = data.labelled & (data.day >= 1) & (data.day <= 35)
    scores = np.zeros(len(test))
    for seed in SEEDS:
        s, _ = fit_predict(data, train_mask, epochs=EPOCHS, seed=seed)
        scores += pd.Series(s[rows]).rank(pct=True).to_numpy()
        print(f"  seed {seed} done", flush=True)
    write_submission(scores / len(SEEDS), "gnn_5seed")
    base = pd.read_csv(SUBMISSIONS / BASE)["target"].rank(pct=True)
    gnn = pd.Series(scores).rank(pct=True)
    print(f"  spearman gnn vs {BASE}: {gnn.corr(base, method='spearman'):.3f}")
    path = write_submission(((1 - WEIGHT) * base + WEIGHT * gnn).to_numpy(), "blend3_gnn20")
    print(f"    -> {path.name}")


if __name__ == "__main__":
    main()
