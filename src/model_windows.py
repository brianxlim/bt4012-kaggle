"""Window bagging: the model of record trained from several start days, rank-averaged.

Blending the full-history model (0.95521) with a days-23-35 model (0.95353) 50/50
scored 0.95629, the first blend in this project to gain. The two rank the test
rows differently (Spearman 0.896) while being nearly as good as each other.
This widens that: one model per start day, each 79 trees x 5 seeds, equal weight.

    python src/model_windows.py 1 12 18 23 28
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

ROUNDS = 79


def main():
    starts = [int(s) for s in sys.argv[1:]] or [1, 12, 18, 23, 28]
    labelled, test = load_labelled(), load_test()
    blended = np.zeros(len(test))
    for start in starts:
        part = labelled[labelled["time_step"] >= start].reset_index(drop=True)
        scores = fit_final(part, test, FEATURES, ROUNDS, TUNED)
        blended += pd.Series(scores).rank(pct=True).to_numpy()
        write_submission(scores, f"window_from{start}")
        print(f"  from day {start:>2}: {len(part):>6,} rows")
    name = "windows_" + "_".join(map(str, starts))
    print(f"    -> {write_submission(blended / len(starts), name).name}")


if __name__ == "__main__":
    main()
