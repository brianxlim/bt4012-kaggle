"""TabPFN: a pretrained tabular foundation model, used without training.

TabPFN reads a sample of labelled rows as context and predicts new rows in one
forward pass. It cannot take all 31k rows on a CPU, so this bags it: each bag
keeps every illicit row (3,644) plus a random draw of licit rows, up to
CONTEXT rows, and the bags' rank-averaged scores are the prediction.

On 3,000 rows from days 20-29 scored on days 30-35 it beat xgboost trained on
the same rows (0.9978 vs 0.9955). Needs TABPFN_TOKEN (licence at priorlabs.ai).

    python src/model_tabpfn.py                 # all days, bags 0-2
    python src/model_tabpfn.py 18 0 3 recent18 # context from day 18 on, bags 0-2
    python src/model_tabpfn.py 1 3 6 all       # all days, bags 3-5
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

from data import FEATURES, SUBMISSIONS, load_labelled, load_test
from experiment import write_submission

CONTEXT = int(os.environ.get("TABPFN_CONTEXT", 8000))  # rows of context per bag
BAGS = 3
ESTIMATORS = int(os.environ.get("TABPFN_ESTIMATORS", 2))  # TabPFN's internal ensemble size
CHUNK = int(os.environ.get("TABPFN_CHUNK", 2000))  # test rows per forward pass
# Pretrained weights, e.g. "v2_6", "v3" or "v3_5", or "tabicl" for TabICLv2 instead of TabPFN;
# unset keeps the package default and the old file names.
VERSION = os.environ.get("TABPFN_VERSION", "")
CHECKPOINT = os.environ.get("TABPFN_CKPT", "")  # a specific checkpoint file of that version
# Encode the context once and reuse it for every test chunk, instead of once per chunk.
CACHED = os.environ.get("TABPFN_CACHED", "") == "1"
# Comma-separated features to leave out, e.g. the time markers feat_100,feat_101,feat_103
DROP = [f for f in os.environ.get("TABPFN_DROP", "").split(",") if f]
USED = [f for f in FEATURES if f not in DROP]


def make_model(bag):
    if VERSION == "tabicl":
        from tabicl import TabICLClassifier
        return TabICLClassifier(n_estimators=ESTIMATORS, device=pick_device(), random_state=bag,
                                batch_size=int(os.environ.get("TABICL_BATCH", 2)))
    kwargs = dict(device=pick_device(), n_estimators=ESTIMATORS, random_state=bag, ignore_pretraining_limits=True)
    if CACHED:
        kwargs.update(fit_mode="fit_with_cache", kv_cache_precision=None)
    if not VERSION:
        return TabPFNClassifier(**kwargs)
    from tabpfn.constants import ModelVersion
    if CHECKPOINT:
        kwargs["model_path"] = CHECKPOINT
    return TabPFNClassifier.create_default_for_version(ModelVersion(VERSION.replace("_", ".")), **kwargs)


def pick_device():
    """TABPFN_DEVICE if set, else an NVIDIA GPU, else Apple Silicon, else CPU."""
    if os.environ.get("TABPFN_DEVICE"):
        return os.environ["TABPFN_DEVICE"]
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_env():
    """TABPFN_TOKEN from .env, so TabPFN never falls back to a browser login."""
    path = Path(__file__).resolve().parents[1] / ".env"
    for line in path.read_text().splitlines() if path.exists() else []:
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main():
    load_env()
    first_day, bag_from, bag_to, tag = 1, 0, BAGS, ""
    if len(sys.argv) > 1:
        first_day, bag_from, bag_to, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    labelled, test = load_labelled(), load_test()
    labelled = labelled[labelled["time_step"] >= first_day].reset_index(drop=True)
    y = labelled["label"].to_numpy()
    illicit, licit = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    Xt = test[USED].to_numpy()
    total = np.zeros(len(test))
    prefix = f"tabpfn_{tag}_" if tag and tag != "all" else "tabpfn_"
    if VERSION:
        family = "tabicl" if VERSION == "tabicl" else f"tabpfn{VERSION.replace('_', '').replace('.', '')}"
        prefix = prefix.replace("tabpfn_", f"{family}_", 1)
    for bag in range(bag_from, bag_to):
        out = SUBMISSIONS.parent / "cache" / f"{prefix}bag{bag}.npy"
        if not out.exists():
            rng = np.random.default_rng(bag)
            take = min(len(licit), CONTEXT - len(illicit))
            rows = np.r_[illicit, rng.choice(licit, take, replace=False)]
            model = make_model(bag)
            model.fit(labelled.loc[rows, USED].to_numpy(), y[rows])
            parts = []
            for i in range(0, len(Xt), CHUNK):
                # each chunk is saved, so a stopped run resumes where it left off
                chunk_path = out.with_name(f"{out.stem}_{i}.npy")
                if not chunk_path.exists():
                    start = time.time()
                    np.save(chunk_path, model.predict_proba(Xt[i:i + CHUNK])[:, 1])
                    print(f"  bag {bag} rows {i}-{i + CHUNK} in {time.time() - start:.0f}s", flush=True)
                parts.append(np.load(chunk_path))
            np.save(out, np.concatenate(parts))
        total += pd.Series(np.load(out)).rank(pct=True).to_numpy()
        print(f"  bag {bag} done", flush=True)
    name = f"{prefix}bags{bag_from}-{bag_to - 1}" if tag else "tabpfn_bag3"
    path = write_submission(total / (bag_to - bag_from), name)
    best = pd.read_csv(SUBMISSIONS / "blend_tabpfn_recent75_12k.csv")["target"]
    print(f"  spearman vs best {pd.Series(total).corr(best, method='spearman'):.3f} -> {path.name}")


if __name__ == "__main__":
    main()
