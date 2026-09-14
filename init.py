"""One-time setup: fetch the competition data into this repo and build the cache.

Run this once after cloning:

    python init.py

This writes the four CSVs into ./data.

Needs a Kaggle API token in .env (never committed):

    KAGGLE_API_TOKEN=KGAT_...

Get one from https://www.kaggle.com/settings -> API -> Create New Token.
"""

import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COMPETITION = "bt-4012-competition-2026"

# name -> expected row count, so a truncated download is caught here rather
# than three steps later in a model
EXPECTED = {
    "train.csv": 141_772,
    "test.csv": 15_329,
    "txs_edgelist.csv": 234_355,
    "sample_submission.csv": 15_329,
}


def load_env() -> None:
    """Read .env into the environment. The kaggle package authenticates on
    import, so this has to happen before that import."""
    path = ROOT / ".env"
    if not path.exists():
        sys.exit(
            f"No {path} found.\n"
            "Create it with a single line:\n\n"
            "    KAGGLE_API_TOKEN=KGAT_your_token_here\n\n"
            "Token from https://www.kaggle.com/settings -> API -> Create New Token."
        )
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    if not os.environ.get("KAGGLE_API_TOKEN"):
        sys.exit("KAGGLE_API_TOKEN is missing from .env")


def download() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    DATA.mkdir(exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"downloading {COMPETITION} into {DATA.relative_to(ROOT)}/ ...")
    api.competition_download_files(COMPETITION, path=str(DATA), quiet=False)

    for archive in DATA.glob("*.zip"):
        print(f"unzipping {archive.name}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(DATA)
        archive.unlink()


def verify() -> bool:
    """True when all four files are present and the right length."""
    ok = True
    for name, rows in EXPECTED.items():
        path = DATA / name
        if not path.exists():
            print(f"  missing  {name}")
            ok = False
            continue
        with path.open() as f:
            found = sum(1 for _ in f) - 1  # minus the header
        mark = "ok      " if found == rows else "WRONG   "
        if found != rows:
            ok = False
        size = path.stat().st_size / 1e6
        print(f"  {mark} {name:24s} {found:>7,} rows  {size:>6.1f} MB")
    return ok


def build_cache() -> None:
    """First load converts each CSV to float32 parquet. 460 MB of CSV takes
    about a minute now and a tenth of a second every time after."""
    sys.path.insert(0, str(ROOT / "src"))
    from data import load_edges, load_test, load_train

    print("\nbuilding the parquet cache (one minute, then instant forever)")
    for name, load in (("train", load_train), ("test", load_test), ("edges", load_edges)):
        print(f"  {name:6s} {len(load()):>7,} rows")


def main() -> None:
    load_env()
    if verify():
        print("data already present, skipping download")
    else:
        download()
        print("\nverifying:")
        if not verify():
            sys.exit("download incomplete — delete data/ and run again")
    build_cache()
    print(f"\nready. next: python src/model_tuned.py")


if __name__ == "__main__":
    main()
