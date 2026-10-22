"""Shared data loading / chunking helpers for the foundation-model runners (run_kumo.py, run_limix.py).

Lives outside the Kaggle repo on purpose. Reads the repo's parquet cache read-only.

Env vars (all optional):
  FM_REPO            repo root (default C:\\Users\\brixn\\Documents\\BT4012\\bt4012-kaggle)
  FM_MAX_TEST        truncate the test set to its first N rows (smoke tests)
  FM_MAX_FEATURES    keep only the K features with highest univariate |AUC-0.5| on the context rows (default: all 165)
  FM_MAX_ILLICIT_FRAC  cap on the illicit share of the context (default 0.5; only bites when #illicit is large)
  FM_CHUNK           query rows per predict call (runner-specific default)
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO = Path(os.environ.get("FM_REPO", r"C:\Users\brixn\Documents\BT4012\bt4012-kaggle"))
CACHE = REPO / "cache"
FEATURES = [f"feat_{i}" for i in range(1, 166)]  # same as src/data.py


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _univariate_auc_order(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Columns sorted by |AUC - 0.5| descending (rank-sum AUC, ties averaged)."""
    from scipy.stats import rankdata

    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    ranks = np.apply_along_axis(rankdata, 0, x)
    auc = (ranks[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.argsort(-np.abs(auc - 0.5), kind="stable")


def load_problem(first_day: int, context_rows: int, seed: int):
    """Returns (Xc, yc, Xt, test_index, feature_names).

    Context = ALL illicit rows (capped at FM_MAX_ILLICIT_FRAC of context_rows) + seeded random licit rows,
    from labelled training rows with time_step >= first_day. Test rows are in data/sample_submission.csv order.
    """
    train = pd.read_parquet(CACHE / "train.parquet", columns=["time_step", "label"] + FEATURES)
    lab = train[train["label"].notna() & (train["time_step"] >= first_day)]
    ill = lab[lab["label"] == 1]
    lic = lab[lab["label"] == 0]
    rng = np.random.default_rng(seed)

    n_ill_all, n_lab = len(ill), len(lab)
    max_ill = int(float(os.environ.get("FM_MAX_ILLICIT_FRAC", "0.5")) * context_rows)
    if len(ill) > max_ill:
        log(f"NOTE: {len(ill)} illicit rows >= cap {max_ill}; subsampling illicit to {max_ill} (seeded)")
        ill = ill.iloc[np.sort(rng.choice(len(ill), size=max_ill, replace=False))]
    n_lic = min(len(lic), max(context_rows - len(ill), 0))
    lic = lic.iloc[np.sort(rng.choice(len(lic), size=n_lic, replace=False))]
    ctx = pd.concat([ill, lic]).sample(frac=1.0, random_state=seed)  # shuffle rows
    log(f"context: {len(ctx)} rows ({int((ctx['label'] == 1).sum())} illicit, {int((ctx['label'] == 0).sum())} licit), "
        f"days >= {first_day} (labelled pool: {n_ill_all} illicit / {n_lab} labelled)")

    test = pd.read_parquet(CACHE / "test.parquet")
    sub = pd.read_csv(REPO / "data" / "sample_submission.csv")
    test = test.set_index("index").loc[sub["index"].to_numpy()].reset_index()  # exact submission order
    max_test = os.environ.get("FM_MAX_TEST")
    if max_test:
        test = test.iloc[: int(max_test)]
        log(f"FM_MAX_TEST -> truncated test to {len(test)} rows")

    feats = FEATURES
    xc = ctx[feats].to_numpy(dtype=np.float32)
    yc = ctx["label"].to_numpy(dtype=np.int64)
    k = os.environ.get("FM_MAX_FEATURES")
    if k and int(k) < len(feats):
        order = _univariate_auc_order(xc, yc)[: int(k)]
        order = np.sort(order)
        feats = [feats[i] for i in order]
        xc = xc[:, order]
        log(f"FM_MAX_FEATURES -> kept top {len(feats)} features by univariate AUC")
    xt = test[feats].to_numpy(dtype=np.float32)
    return xc, yc, xt, test["index"].to_numpy(), feats


def predict_in_chunks(predict_fn, xt: np.ndarray, chunk: int, out_csv: Path, test_index: np.ndarray):
    """predict_fn(x_chunk) -> P(illicit) array. Resumable: per-chunk .npy parts under <out_csv>.parts/."""
    out_csv = Path(out_csv)
    parts = out_csv.with_suffix(out_csv.suffix + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    preds = []
    t_all = time.time()
    for s in range(0, len(xt), chunk):
        f = parts / f"part_{s}.npy"
        if f.exists():
            p = np.load(f)
            log(f"chunk {s}: loaded existing part")
        else:
            t0 = time.time()
            p = np.asarray(predict_fn(xt[s : s + chunk]), dtype=np.float64)
            np.save(f, p)
            log(f"chunk {s}-{min(s + chunk, len(xt))}: {time.time() - t0:.1f}s")
        preds.append(p)
    p = np.concatenate(preds)
    assert p.shape == (len(xt),) and np.isfinite(p).all(), (p.shape, np.isnan(p).sum())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": test_index, "target": p}).to_csv(out_csv, index=False)
    log(f"wrote {out_csv} ({len(p)} rows) in {time.time() - t_all:.1f}s; mean P(illicit)={p.mean():.4f}")
    return p


def parse_cli(argv):
    if len(argv) < 5:
        sys.exit("usage: python run_X.py <first_day> <context_rows> <seed> <out_csv> [device]")
    first_day, context_rows, seed = int(argv[1]), int(argv[2]), int(argv[3])
    out_csv = Path(argv[4])
    device = argv[5] if len(argv) > 5 else "cuda"
    return first_day, context_rows, seed, out_csv, device
