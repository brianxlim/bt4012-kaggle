"""Best blend rebuilt with two labelled-neighbour features added to every model.

g_nlab: how many payment neighbours carry a label (train) or a test.csv row
(test). test.csv holds exactly the labelled day-36-49 transactions, so the
definition is identical on both sides. g_frac: that count over total degree.

Alone g_nlab separates the classes at AUC 0.68 (no labelled neighbours: 24%
illicit). Offline, added to xgboost: +0.0025 on seeds 0-4, +0.0004 on seeds
5-14, i.e. noise. But offline has misjudged every feature change, so this is a
leaderboard test. Same three windows and weights as the 0.95646 blend.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from data import FEATURES, load_labelled, load_test
from experiment import write_submission
from graph_labelled import graph_feats
from model_tuned import TUNED
from train import fit_final
from validation import assert_no_shift

EXTRA = ["g_nlab", "g_frac"]
WINDOWS = [(1, 0.5), (23, 0.25), (18, 0.25)]


def main():
    g = graph_feats()[EXTRA]
    labelled = load_labelled().join(g, on="txId")
    test = load_test().join(g, on="txId")
    print(assert_no_shift(labelled, test, EXTRA).to_string(index=False))
    columns = FEATURES + EXTRA
    blended = np.zeros(len(test))
    for start, weight in WINDOWS:
        part = labelled[labelled["time_step"] >= start].reset_index(drop=True)
        scores = fit_final(part, test, columns, 79, TUNED)
        blended += weight * pd.Series(scores).rank(pct=True).to_numpy()
    print(f"    -> {write_submission(blended, 'blend3_nlab').name}")


if __name__ == "__main__":
    main()
