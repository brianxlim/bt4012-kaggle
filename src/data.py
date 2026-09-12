"""Loading and caching for the BT4012 / Elliptic transaction data.

The raw train.csv is 460 MB, so on first use each file is converted to a
float32 parquet cache. Subsequent loads take about a second instead of a minute.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"

# feat_1..feat_93 describe the transaction itself; feat_94..feat_165 are the
# organisers' pre-computed summaries of each transaction's immediate neighbours.
FEATURES = [f"feat_{i}" for i in range(1, 166)]
LOCAL_FEATURES = [f"feat_{i}" for i in range(1, 94)]
NEIGHBOUR_FEATURES = [f"feat_{i}" for i in range(94, 166)]

# Days 1-35 are labelled and given to us; days 36-49 are what we must predict.
TRAIN_STEPS = (1, 35)
TEST_STEPS = (36, 49)


def _cached(name: str, build) -> pd.DataFrame:
    path = CACHE / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frame = build()
    CACHE.mkdir(exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def _read(filename: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / filename)
    frame[FEATURES] = frame[FEATURES].astype("float32")
    frame["time_step"] = frame["time_step"].astype("int16")
    frame["txId"] = frame["txId"].astype("int64")
    return frame


def load_train() -> pd.DataFrame:
    """All 141,772 training rows, including the 110,537 with no label."""
    return _cached("train", lambda: _read("train.csv"))


def load_test() -> pd.DataFrame:
    """The 15,329 rows to predict, in submission order."""
    return _cached("test", lambda: _read("test.csv"))


def load_edges() -> pd.DataFrame:
    """Payment links between transactions. Covers 46,668 nodes we have no features for."""
    return _cached("edges", lambda: pd.read_csv(DATA / "txs_edgelist.csv"))


def load_labelled() -> pd.DataFrame:
    """Training rows that carry a fraud label: 27,591 legitimate, 3,644 illicit."""
    train = load_train()
    labelled = train[train["label"].notna()].copy()
    labelled["label"] = labelled["label"].astype("int8")
    return labelled.reset_index(drop=True)
