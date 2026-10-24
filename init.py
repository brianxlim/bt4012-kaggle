"""One-time setup: fetch the competition data into this repo and build the cache.

Run this once after cloning, with the repo's virtual environment:

    .venv\\Scripts\\python init.py        (macOS/Linux: .venv/bin/python init.py)

This writes the four CSVs into ./data. If they are already there, nothing is
downloaded and no token is needed.

The download needs a Kaggle account that has joined the competition, and an
API token in .env (never committed):

    KAGGLE_API_TOKEN=KGAT_...

An exported KAGGLE_API_TOKEN or a token in ~/.kaggle works too. Get one from
https://www.kaggle.com/settings -> API -> Create New Token. Without a token,
download the four CSVs from the competition's Data tab into data/ instead.
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
    import, so this has to happen before that import. A missing .env is fine
    when the token is already exported or saved in ~/.kaggle."""
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    saved = [Path.home() / ".kaggle" / name for name in ("access_token", "access_token.txt", "kaggle.json")]
    if not os.environ.get("KAGGLE_API_TOKEN") and not any(p.exists() for p in saved):
        sys.exit(
            "No Kaggle token found in .env, the environment or ~/.kaggle.\n"
            f"Create {path} with a single line:\n\n"
            "    KAGGLE_API_TOKEN=KGAT_your_token_here\n\n"
            "Token from https://www.kaggle.com/settings -> API -> Create New Token.\n"
            f"Or download the four CSVs from https://www.kaggle.com/competitions/{COMPETITION}/data into data/."
        )


def download() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    DATA.mkdir(exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"downloading {COMPETITION} into {DATA.relative_to(ROOT)}/ ...")
    try:
        api.competition_download_files(COMPETITION, path=str(DATA), quiet=False)
    except Exception as err:  # most often: the account has not joined the competition
        sys.exit(f"download failed: {err}\nJoin the competition and accept its rules at "
                 f"https://www.kaggle.com/competitions/{COMPETITION}/rules, check the token, then run again.")

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
    if verify():
        print("data already present, skipping download")
    else:
        load_env()  # only a download needs the token
        download()
        print("\nverifying:")
        if not verify():
            sys.exit("download incomplete — delete data/ and run again")
    build_cache()
    print("\nready. next: run e1122376.ipynb (README, Running the notebook)")


if __name__ == "__main__":
    main()
