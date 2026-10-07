"""EXPERIMENT: model of record trained only on days 23-35, past the regime cliff.

Days 1-22 behave differently from day 23 onward (progress.md, section 4.1), and
the test days follow the later regime. Offline this is flat (0.9970 vs 0.9973 on
the saturated post-cliff windows), and the pre-cliff folds cannot score it at
all, so the leaderboard is the only judge. Settings and tree count unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data import FEATURES, load_labelled, load_test
from experiment import write_submission
from model_tuned import TUNED
from train import fit_final

FIRST_DAY = 23
ROUNDS = 79


def main():
    labelled, test = load_labelled(), load_test()
    recent = labelled[labelled["time_step"] >= FIRST_DAY].reset_index(drop=True)
    print(f"  training on days {FIRST_DAY}-35: {len(recent):,} of {len(labelled):,} rows")
    scores = fit_final(recent, test, FEATURES, ROUNDS, TUNED)
    print(f"    -> {write_submission(scores, 'recent_d23').name}")


if __name__ == "__main__":
    main()
