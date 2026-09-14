# BT4012 — Fraudulent Bitcoin Transaction Detection

Kaggle competition [`bt-4012-competition-2026`](https://www.kaggle.com/competitions/bt-4012-competition-2026).
Predict which Bitcoin transactions are illicit, scored by ROC AUC.

---

## Setup

> Pre-requisite: Python 3.12 and a Kaggle account.

**1. Virtual environment and dependencies**

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

XGBoost and LightGBM need OpenMP on macOS:

```bash
brew install libomp
```

**2. Kaggle API token**

Go to <https://www.kaggle.com/settings> → API → **Create New Token**. Put it in
a `.env` file at the repo root:

```
KAGGLE_API_TOKEN=KGAT_your_token_here
```

`.env` is gitignored. Never commit it.

**3. Fetch the data and build the cache**

```bash
python init.py
```

This downloads the four CSVs into `data/`, checks each one has the expected
number of rows, and converts them to parquet in `cache/`. The raw `train.csv` is
461 MB and takes about a minute to parse; every load after that is instant.
Both `data/` and `cache/` are gitignored — the data is 515 MB, well over
GitHub's limit.

Re-running `init.py` is safe. It skips the download when the files are
already correct.

---

## Reproducing the submission

```bash
python src/model_tuned.py
```

Takes a few minutes. It scores the model over 3 time folds × 5 seeds, prints the
result with an error bar, then fits on all labelled days and writes
`submissions/spw757_mcw8.csv` and a row in `reports/experiments.csv`.

Then submit it:

```bash
.venv/bin/kaggle competitions submit -c bt-4012-competition-2026 -f submissions/spw757_mcw8.csv -m "describe the change"
```

**Three submissions per day.** Read [`CONTEXT.md`](CONTEXT.md) before spending
one — most ideas that look good offline have lost ground on the leaderboard, and
that document records which kinds do and do not transfer.

---

## What the model is

Gradient-boosted decision trees (XGBoost) on the 165 supplied features, labelled
rows only, five random seeds rank-averaged.

| Setting | Value | Why |
|---|---|---|
| `max_depth` | 10 | |
| `colsample_bytree` | 0.5 | |
| `learning_rate` | 0.1 | |
| `scale_pos_weight` | 7.57 | Only 11.67% of rows are illicit; 27,591 / 3,644 makes both classes weigh the same |
| `min_child_weight` | 8 | Stops the tree carving out the tiny illicit clusters that the weighting makes look important |

The last two are the difference between 0.94812 and 0.95333. Neither works
alone — see [`reports/progress.md`](reports/progress.md).

---

## Layout

```
init.py                 one-time setup: download data, build cache
src/data.py             loading and parquet caching
src/validation.py       time-aware folds and the train/test drift check
src/train.py            the shared training routine every experiment uses
src/model_tuned.py      the model of record
src/tune.py             settings sweep
src/experiment.py       writing submissions and logging results
reports/experiments.csv one row per submission, offline score against leaderboard
reports/progress.md     what has been tried, what worked, what did not
reports/tuning.csv      the settings sweep
CONTEXT.md              what future work needs to know before touching anything
ADR.md                  plain-English glossary of every term used
data/  cache/  .env     gitignored
```

---

## Competition facts

- Elliptic Bitcoin dataset: 203,769 transactions over 49 days, 234,355 payment links.
- Train is days 1–35, test is days 36–49. **The split is by time**, so random
  cross-validation gives a misleading answer.
- `train.csv` holds every transaction for days 1–35, labelled or not.
  `test.csv` holds only the labelled ones for days 36–49. That asymmetry has
  already cost one submission — see `CONTEXT.md`.
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
