"""TabPFN: a pretrained tabular foundation model, used without training.

TabPFN reads a sample of labelled rows as context and predicts new rows in one
forward pass. It cannot take all 31k rows on a CPU, so this bags it: each bag
keeps every illicit row (3,644) plus a random draw of licit rows, up to
CONTEXT rows, and the bags' rank-averaged scores are the prediction.

On 3,000 rows from days 20-29 scored on days 30-35 it beat xgboost trained on
the same rows (0.9978 vs 0.9955). Needs TABPFN_TOKEN (licence at priorlabs.ai).

    python src/model_tabpfn.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

from data import FEATURES, SUBMISSIONS, load_labelled, load_test
from experiment import write_submission

CONTEXT = 8000
BAGS = 3
ESTIMATORS = 2
CHUNK = 2000


def main():
    labelled, test = load_labelled(), load_test()
    y = labelled["label"].to_numpy()
    illicit, licit = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    Xt = test[FEATURES].to_numpy()
    total = np.zeros(len(test))
    for bag in range(BAGS):
        out = SUBMISSIONS.parent / "cache" / f"tabpfn_bag{bag}.npy"
        if not out.exists():
            rng = np.random.default_rng(bag)
            rows = np.r_[illicit, rng.choice(licit, CONTEXT - len(illicit), replace=False)]
            model = TabPFNClassifier(device="cpu", n_estimators=ESTIMATORS, random_state=bag,
                                     ignore_pretraining_limits=True)
            model.fit(labelled.loc[rows, FEATURES].to_numpy(), y[rows])
            parts = []
            for i in range(0, len(Xt), CHUNK):
                # each chunk is saved, so a stopped run resumes where it left off
                chunk_path = out.with_name(f"tabpfn_bag{bag}_{i}.npy")
                if not chunk_path.exists():
                    start = time.time()
                    np.save(chunk_path, model.predict_proba(Xt[i:i + CHUNK])[:, 1])
                    print(f"  bag {bag} rows {i}-{i + CHUNK} in {time.time() - start:.0f}s", flush=True)
                parts.append(np.load(chunk_path))
            np.save(out, np.concatenate(parts))
        total += pd.Series(np.load(out)).rank(pct=True).to_numpy()
        print(f"  bag {bag} done", flush=True)
    path = write_submission(total / BAGS, "tabpfn_bag3")
    best = pd.read_csv(SUBMISSIONS / "blend4_full40_d23_d18_d12.csv")["target"]
    print(f"  spearman vs best {pd.Series(total).corr(best, method='spearman'):.3f} -> {path.name}")


if __name__ == "__main__":
    main()
