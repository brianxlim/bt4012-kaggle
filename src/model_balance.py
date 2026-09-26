"""Model of record with heavier class balancing. Candidate after 0.95521.

Same tree shape as model_tuned.py; only the class-balance pair moves further
along the direction that produced the +0.0052 leaderboard gain:

    scale_pos_weight  7.57 -> 24   each illicit row counts as 24 licit ones
    min_child_weight  8    -> 16   each split needs twice the evidence

Offline, paired against the model of record over the 3 pre-cliff folds:
a 3x3 grid of spw {12,16,24} x mcw {12,16,24} was positive in every cell
(+0.0013 to +0.0073, seeds 20-29). The top cell re-measured on seeds 30-39,
which the grid never saw, at +0.0071 (3/3 folds up) - no shrinkage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data import FEATURES, load_labelled, load_test
from model_tuned import TUNED
from train import run

BALANCED = {**TUNED, "scale_pos_weight": 24, "min_child_weight": 16}


def main():
    labelled, test = load_labelled(), load_test()
    run("balance_spw24_mcw16", labelled, test, FEATURES, params=BALANCED)


if __name__ == "__main__":
    main()
