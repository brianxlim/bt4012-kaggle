"""Best 4-way window blend rebuilt with fewer trees per model.

Every change that boosted harder has lost on the leaderboard: more trees at a
lower learning rate (-0.0095), a 400-tree config (-0.0095), LightGBM at 300
trees blended 50/50 (-0.0033). The model of record uses 79. If less boosting
transfers better to the shifted test days, fewer trees should help.

    python src/model_fewtrees.py 40
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from data import FEATURES, load_labelled, load_test
from experiment import write_submission
from model_tuned import TUNED
from train import fit_final

WINDOWS = [(1, 0.4), (23, 0.2), (18, 0.2), (12, 0.2)]  # the 0.95672 blend


def main():
    rounds = int(sys.argv[1])
    labelled, test = load_labelled(), load_test()
    blended = np.zeros(len(test))
    for start, weight in WINDOWS:
        part = labelled[labelled["time_step"] >= start].reset_index(drop=True)
        scores = fit_final(part, test, FEATURES, rounds, TUNED)
        blended += weight * pd.Series(scores).rank(pct=True).to_numpy()
    print(f"    -> {write_submission(blended, f'blend4_trees{rounds}').name}")


if __name__ == "__main__":
    main()
