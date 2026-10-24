# BT4012 — Fraudulent Bitcoin Transaction Detection

Kaggle competition [`bt-4012-competition-2026`](https://www.kaggle.com/competitions/bt-4012-competition-2026).
Predict which Bitcoin transactions are illicit, scored by ROC AUC.

**Model of record:** a TabPFN v3.5 blend (.45) plus NVIDIA Kumo Tabular (.55),
`submissions/blend_tabpfn_kumo22_55.csv`. Public score **0.97357**, 3rd on the
last day (1st 0.97582, 2nd 0.97446). The two final selections on Kaggle are this
blend and the TabPFN blend alone, `submissions/blend_tabpfn_recent75_12k.csv`
(0.96825). The first Kumo blend, TabPFN .75 + Kumo .25, scored **0.97162**: 2nd
when submitted on 2026-10-23, 3rd that evening (2nd 0.97190, leader 0.97446).

`submissions/` keeps only the 29 files of the final model. Earlier submission
files and the experiment scripts that did not reach it were removed on 2026-10-24;
they are in the git history at commit `d93e014`.

---

## Setup

Pre-requisites: Python, a Kaggle account and an NVIDIA GPU (built on an RTX 3060 Ti,
8 GB). One 8k-context TabPFN bag took about 18 minutes on 4 CPU cores and about 3.5
on the GPU. Commands are PowerShell from the repo root; on macOS or Linux use
`.venv/bin/python` (and `brew install libomp` for XGBoost and LightGBM).

**1. Environment.** The pins in `requirements.txt` are the versions of the final runs
(Python 3.11). Reuse the system CUDA torch (2.5.1+cu124):

If the machine has no CUDA torch yet, install it first:
`python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124`.

```powershell
python -m venv .venv --system-site-packages
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

**2. Kaggle token.** At <https://www.kaggle.com/settings> → API → **Create New
Token**. Put it in `.env` at the repo root as `KAGGLE_API_TOKEN=KGAT_...`
(gitignored, never commit it).

**3. Data.** `python init.py` downloads the four CSVs into `data/`, checks their
row counts and builds the parquet cache in `cache/`. Both are gitignored. Re-running is safe.

**4. TabPFN token.** Add `TABPFN_TOKEN=...` (from priorlabs.ai) to `.env`. The
default weights, TabPFN v3.5, download on the first run. The v3, v2.6 and v2.5
weights also need a one-time licence acceptance at
<https://ux.priorlabs.ai/account/licenses>; the final model does not use them.

**5. Kumo environment (only for the Kumo runs).** Kumo needs a newer torch, so it
gets its own venv, `.venv-fm` (gitignored):

```powershell
python -m venv .venv-fm
.venv-fm\Scripts\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv-fm\Scripts\python -m pip install -r requirements-fm.txt
```

The first run downloads the weights from Hugging Face (`nvidia/Kumo-Tabular`,
revision v1.0.1, `large`, about 855 MB; OpenMDW 1.1 licence, not gated).

---

## Reproducing the submission

**Notebook:** `e1122376.ipynb` documents the whole pipeline in one place. Run it from the repo root with the repo's `.venv`:

```powershell
.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 e1122376.ipynb
```

Do not use `jupyter nbconvert` or `python -m jupyter`: they start the system Python, which has no xgboost. By default the notebook
runs the validation on the GPU (about 10 minutes the first time, cached afterwards; `$env:E1122376_SKIP_VALIDATION="1"` skips it
and needs no GPU), then **re-blends the committed per-run prediction files** and checks the three blends against the committed ones exactly.
That exact match checks the blending arithmetic and the file format, not the models. `$env:E1122376_RERUN="1"` regenerates every TabPFN bag
and Kumo run from `train.csv` and `test.csv` (about 2 hours; the 31k-row Kumo run alone takes about 32 minutes). It is not bit-identical to the
committed files, so it reports rank correlations. The complete rerun ran end to end on 24 October 2026 in 121 minutes:
all 22 Kumo runs matched the committed files byte for byte, the four TabPFN files agreed at Spearman
0.989–0.999, and the final submission at Spearman 0.9996.
A full rerun needs `--ExecutePreprocessor.timeout=-1` (the timeout is per cell) and should write to its own file so it does not overwrite the notebook's saved outputs:

```powershell
$env:E1122376_RERUN="1"; .venv\Scripts\python.exe -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir rebuild --output e1122376_rerun.ipynb e1122376.ipynb
```

Close other GPU programs first: the largest runs use almost all of the 8 GB. After an aborted rerun, check `nvidia-smi` for a leftover Kumo process before resuming.

All bag and blend files are committed in `submissions/`, so step 4 alone rebuilds
the blend. Activate `.venv` first (`.venv\Scripts\Activate.ps1`). On the RTX 3060 Ti
a v3.5 bag takes about 3.5 minutes at 8,000 context rows and 5 to 8 at 12,000; a
Kumo run takes 2 to 3 (about 32 at 31k rows). Runs save each chunk of test rows and
resume if stopped. Saved bags are keyed by version, tag and bag number only, so use
a new tag when you change any other setting.

**1. TabPFN bags** (arguments: `first_day bag_from bag_to tag`)

```powershell
python src/model_tabpfn.py 1 0 6 all                                          # all days, 8k, 6 bags
$env:TABPFN_CONTEXT=12000; python src/model_tabpfn.py 1 0 1 ctx12k            # all days, 12k, 1 bag
python src/model_tabpfn.py 18 0 3 recent18                                    # days 18+, 8k, 3 bags
$env:TABPFN_CONTEXT=12000; python src/model_tabpfn.py 18 0 1 recent18ctx12k   # days 18+, 12k, 1 bag
Remove-Item Env:TABPFN_CONTEXT
```

The blend uses file names from earlier versions of the script. A fresh run writes
`tabpfn_bags0-5.csv`, `tabpfn_ctx12k_bags0-0.csv` and `tabpfn_recent18ctx12k_bags0-0.csv`
instead of `tabpfn_all_bags0-5.csv`, `tabpfn_ctx12k_bag0.csv` and
`tabpfn_recent18ctx12k_bag0.csv` (`recent18` is unchanged), and does not overwrite
the committed files. Rename them or pass the new names in step 2. The committed
bags are already v3.5: a GPU rebuild of the days-18+ 8k bags correlates 0.997
(Spearman) with them.

**2. Best TabPFN blend** (public 0.96825)

```powershell
python src/blend.py blend_tabpfn_recent75_12k.csv tabpfn_all_bags0-5.csv:0.167 tabpfn_ctx12k_bag0.csv:0.083 tabpfn_recent18_bags0-2.csv:0.40 tabpfn_recent18ctx12k_bag0.csv:0.35
```

`blend.py` takes file names inside `submissions/` and rank-averages them, dividing by the sum of the weights.

**3. Kumo runs** (`.venv-fm`, from `src/fm`; arguments: `first_day context_rows seed out_csv device`)

```powershell
cd src/fm
$py = "..\..\.venv-fm\Scripts\python.exe"
foreach ($s in 0..9) { & $py run_kumo.py 18 8000 $s "..\..\submissions\kumo_recent18_s${s}.csv" cuda }
& $py run_kumo.py 18 14380 0 ..\..\submissions\kumo_recent18full_s0.csv cuda
foreach ($s in 0..9) { & $py run_kumo.py 1 12000 $s "..\..\submissions\kumo_all12k_s${s}.csv" cuda }
& $py run_kumo.py 1 31235 0 ..\..\submissions\kumo_allfull_s0.csv cuda
cd ..\..
```

Chunks go to `<out_csv>.parts/` (untracked). Delete that folder before re-running
with a different context or seed under the same output name.

**4. Kumo blend, final blend, submit** (repo root)

```powershell
$r = @(0..9 | ForEach-Object { "kumo_recent18_s$_.csv:0.06818" }) + "kumo_recent18full_s0.csv:0.06818"
$a = @(0..9 | ForEach-Object { "kumo_all12k_s$_.csv:0.02273" }) + "kumo_allfull_s0.csv:0.02273"
python src/blend.py kumo22_blend.csv @r @a
python src/blend.py blend_tabpfn_kumo22_55.csv blend_tabpfn_recent75_12k.csv:0.45 kumo22_blend.csv:0.55
.venv\Scripts\kaggle competitions submit -c bt-4012-competition-2026 -f submissions/blend_tabpfn_kumo22_55.csv -m "describe the change"
```

The Kumo part uses all 22 runs from step 3. The first Kumo blend (0.97162) used
seeds 0–4 only (12 runs) at weight .25. Only the .55 file is kept.

| Kumo weight | Kumo runs | Public score |
| ----------- | --------- | ------------ |
| .25         | 12        | 0.97162      |
| .33         | 22        | 0.97229      |
| .40         | 22        | 0.97280      |
| .55         | 22        | **0.97357**  |

**Three submissions per day**, up to two final selections.

---

## What the model is

TabPFN and Kumo Tabular are pretrained tabular foundation models. Neither is
trained here: each reads a context of labelled rows and predicts the test rows in
one forward pass. The context recipe is the same everywhere: all illicit rows from
a first day on, plus seeded random licit rows up to the context size. Bags differ
in seed, first day and context size, and are rank-averaged. A TabPFN bag uses 2
internal estimators, a Kumo run 8.

| Part             | Runs                 | Context rows         | First day | Weight                     |
| ---------------- | -------------------- | -------------------- | --------- | -------------------------- |
| TabPFN, all days | 6 bags               | 8,000                | 1         | .167                       |
| TabPFN, all days | 1 bag                | 12,000               | 1         | .083                       |
| TabPFN, days 18+ | 3 bags               | 8,000                | 18        | .40                        |
| TabPFN, days 18+ | 1 bag                | 12,000               | 18        | .35                        |
| Kumo, days 18+   | 10 seeds, plus 1 run | 8,000; 14,380 (all)  | 18        | .06818 each (.75 in total) |
| Kumo, all days   | 10 seeds, plus 1 run | 12,000; 31,235 (all) | 1         | .02273 each (.25 in total) |

Weights are within the TabPFN blend and the Kumo blend; the final score is the
TabPFN blend × .45 plus the Kumo blend × .55. Public score: tuned XGBoost 0.95521,
TabPFN alone 0.96026, best TabPFN blend 0.96825, plus Kumo **0.97357** (the first
Kumo blend, 12 runs at weight .25, scored 0.97162). Kumo is
the first non-TabPFN model that added to TabPFN: its seeds agree at Spearman 0.98,
but the averaged Kumo ranks test rows differently from the TabPFN blend (0.867).

Tried against the 0.96825 blend, no gain (public score): TabICLv2 at .20 / .35
0.96799 / 0.96629; three TabPFN v3 checkpoints at .25 0.96701; days-43+ block
offset −0.05 / +0.05 / −0.02 0.96714 / 0.95519 / 0.96805;
8 estimators per bag 0.96694; leaving out the six time-marker features 0.96402.
LimiX-2 was dropped (no test chunk finished in 20+ minutes). More in
[`reports/progress.md`](reports/progress.md).

### Settings

`src/model_tabpfn.py` environment variables, all optional:

| Variable            | Default | Meaning                                                                                                                                                                                            |
| ------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TABPFN_CONTEXT`    | 8000    | Context rows per bag; 99999 = all labelled rows from the first day                                                                                                                                 |
| `TABPFN_ESTIMATORS` | 2       | Internal ensemble size                                                                                                                                                                             |
| `TABPFN_CHUNK`      | 2000    | Test rows per predict call                                                                                                                                                                         |
| `TABPFN_VERSION`    | unset   | `v2_5`, `v2_6`, `v3`, `v3_5` or `tabicl` (TabICLv2). Unset = package default (v3.5) and the old file names. Output: `tabpfn{version}_{tag}_bags{a}-{b}.csv`, e.g. `tabpfnv35_recent18_bags0-2.csv` |
| `TABPFN_CKPT`       | unset   | A specific checkpoint file of that version                                                                                                                                                         |
| `TABPFN_CACHED`     | unset   | `1` encodes the context once (`fit_with_cache`); runs out of memory at 14k context rows or more on 8 GB                                                                                            |
| `TABPFN_DROP`       | unset   | Comma-separated features to leave out                                                                                                                                                              |
| `TABPFN_CLIP`       | unset   | `1` clips test features to the bag's context range                                                                                                                                                 |
| `TABICL_BATCH`      | 2       | Batch size for TabICLv2                                                                                                                                                                            |
| `TABPFN_DEVICE`     | auto    | `cuda`, `mps` or `cpu`; default is CUDA, then Apple MPS, then CPU                                                                                                                                  |
| `TABPFN_TOKEN`      | `.env`  | Prior Labs token                                                                                                                                                                                   |

`src/fm/run_kumo.py`: `KUMO_SIZE` (`large`), `KUMO_ESTIMATORS` (8), `FM_CHUNK` (1000
test rows per call), `FM_REPO` (repo root), `FM_MAX_ILLICIT_FRAC` (0.5, cap on the
illicit share of the context), `FM_MAX_TEST` (first N test rows, for smoke tests).

---

## Layout

```
e1122376.ipynb           the notebook: data checks, feature-engineering evidence, model, validation, submission
init.py                  one-time setup: download data, build cache
requirements.txt         main environment (Python 3.11 pins); requirements-fm.txt is .venv-fm
src/data.py              loading and parquet caching
src/model_tabpfn.py      TabPFN and TabICLv2 bags (TabPFN part of the model of record)
src/fm/run_kumo.py       Kumo Tabular runner (Kumo part of the model of record)
src/fm/fm_common.py      context sampling and chunked prediction for src/fm
src/blend.py             rank-average prediction files with weights
src/train.py validation.py experiment.py   xgboost training, time-aware folds, writing submissions
src/model_tuned.py       tuned xgboost, the earlier model of record (0.95521)
reports/experiments.csv  one row per submission, offline score against leaderboard
reports/progress.md      what has been tried, what worked, what did not
submissions/             the 29 prediction files of the final model (index,target); *.csv.parts/ are untracked Kumo chunks
data/  cache/  .env  .venv/  .venv-fm/    gitignored (cache/ also holds saved TabPFN bags)
```

---

## Competition facts

- Elliptic Bitcoin dataset: 203,769 transactions over 49 days, 234,355 payment links.
- Train is days 1–35, test is days 36–49. **The split is by time**, so random
  cross-validation gives a misleading answer.
- `train.csv` holds every transaction for days 1–35, labelled or not.
  `test.csv` holds only the labelled ones for days 36–49. That asymmetry has
  already cost one submission (`reports/progress.md`, section 7).
- 31,235 labelled training rows: 27,591 licit, 3,644 illicit (11.67%).
- Submission format is `index,target`: 15,329 rows in `test.csv` order, where
  target is the probability a transaction is illicit.

---

## Final report requirements

The deliverable is `e1122376.pdf`, built from the sections below.

### 1. Introduction

Briefly introduce the competition's objective and its relevance to fraud analytics. Explain the nature of the fraudulent Bitcoin transaction detection problem and why it's important. Outline your high-level approach, goals, and key strategies. Aim to provide enough background for readers to grasp the context and significance of the task.

### 2. Methodology

This is the main section of your report. Organize it into the following subsections:

#### 2.1 Data Preprocessing and Feature Engineering

Describe how you cleaned, transformed, and prepared the data. Include your decisions around handling missing values, engineering graph-based features (e.g. node degree, neighborhood aggregates), feature selection/extraction, and any augmentation techniques used. Justify your choices using domain insights or experimental evidence.

#### 2.2 Model Development and Validation

Explain the models you implemented, how you trained them, and how you validated their performance. Mention techniques like cross-validation, hyperparameter tuning, and how you handled class imbalance or overfitting. Clearly describe your evaluation approach.

#### 2.3 Model Exploration and Final Strategy

Summarize the range of models you explored (e.g., logistic regression, decision trees, ensemble methods, graph neural networks) and how you compared them. Discuss why you selected your final submission model(s), referencing performance metrics, validation results, and insights from experimentation.

### 3. Results and Discussion

Present the performance of your model(s) using appropriate metrics such as accuracy, precision, recall, F1-score, or ROC AUC. Include tables or charts where helpful. Discuss the strengths and limitations of your approach and reflect on possible causes for model behavior.

### 4. Conclusion and Reflection

Summarize your main findings and what you learned from the project. Reflect on what worked well, what could be improved, and how this experience informs your understanding of AI-driven fraud detection
