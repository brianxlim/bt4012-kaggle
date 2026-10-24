# Progress report

BT4012 Kaggle competition `bt-4012-competition-2026` — detecting illicit Bitcoin
transactions in the Elliptic dataset, scored by ROC AUC.

**Standing: 3rd place, public score 0.97357** (2026-10-24, final day; leader
0.97582, 2nd 0.97446). 70 submissions. Model of record: TabPFN blend .45 +
NVIDIA Kumo Tabular .55. The xgboost model below was 1st on 2026-09-17 at 0.95521.
Last updated 2026-10-24.

This is the running log, kept as written. On 2026-10-24 the repo was trimmed to
the final model: `submissions/` keeps its 29 files and `src/` keeps nine scripts.
Other submission files, scripts (`src/model_alt.py`, `src/tune_v2.py`, ...) and
`reports/tuning*.csv` named below are in the git history at commit `d93e014`.

---

## 1. Result

Standing on 2026-10-24, public leaderboard:

| | Public score |
|---|---|
| Leader | 0.97582 |
| 2nd place | 0.97446 |
| **This model (TabPFN .45 + Kumo .55)** | **0.97357** |
| TabPFN .75 + Kumo .25 (2026-10-23) | 0.97162 |
| Previous model of record (TabPFN-only blend) | 0.96825 |
| Earlier model of record (xgboost, 2026-09-17) | 0.95521 |
| Organisers' baseline | 0.92119 |

The model of record is `submissions/blend_tabpfn_kumo22_55.csv`, a rank-average
of two parts. Both are pretrained tabular models used in-context, with no
training on this data: they read a sample of labelled rows and score the test rows.

| Part | Weight | What it is |
|---|---|---|
| Best TabPFN blend (`blend_tabpfn_recent75_12k.csv`) | .45 | TabPFN v3.5 bags: all-days 8k × 6 bags .167, all-days 12k .083, days-18+ 8k × 3 bags .40, days-18+ 12k .35 |
| Kumo component (`kumo22_blend.csv`) | .55 | NVIDIA Kumo Tabular large, 22 runs: days-18+ 8k × 10 and 14k (.75), all-days 12k × 10 and 31k (.25) |

It scored 0.97357, +0.0053 over the TabPFN-only blend. §27 has the first Kumo
result, §29 the weight search on the last day.

On 2026-09-17 the model of record was the xgboost model below, and it was 1st:

| | Public score, 2026-09-17 |
|---|---|
| **xgboost model** | **0.95521** |
| 2nd place | 0.95357 |
| 3rd place | 0.95085 |
| Organisers' baseline | 0.92119 |

That earlier model is gradient-boosted decision trees (XGBoost) on the 165
supplied features, labelled rows only, five random seeds rank-averaged.
`src/model_tuned.py`.

| Setting | Value |
|---|---|
| `max_depth` | 12 |
| `colsample_bytree` | 0.7 |
| `subsample` | 1.0 |
| `learning_rate` | 0.1 |
| `scale_pos_weight` | 7.57 |
| `min_child_weight` | 8 |
| `gamma` | 0.2 |
| trees | 79 |
| seeds | 5, rank-averaged |

No feature engineering, no external data, no graph features, no use of the
110,537 unlabelled rows. Every one of those was tried and none helped. The
current model adds none of them either. It uses the 165 supplied features and the
labelled rows; the only outside ingredient is the pretrained weights.

---

## 2. Every submission

| # | Change | Offline | Leaderboard | Δ leaderboard |
|---|---|---|---|---|
| 1 | Baseline XGBoost, raw 165 features | 0.8759 | 0.93902 | — |
| 2 | 5-seed rank averaging | 0.8917 | 0.94201 | +0.0030 |
| 3 | Tuned depth 10, colsample 0.5, lr 0.1 | 0.9030 | 0.94812 | +0.0061 |
| 4 | + 22 graph features, 72 neighbourhood only | 0.9465 | 0.89413 | **−0.0540** |
| 5 | Dropped 21 distribution-shifted features | 0.9109 | 0.93984 | **−0.0083** |
| 6 | Half neural net, half XGBoost | 0.9277 | 0.94800 | −0.0001 |
| 7 | + `scale_pos_weight` 7.57, `min_child_weight` 8 | 0.9101 | 0.95333 | **+0.0052** |
| 8 | + tree-shape ridge: d12, cs 0.7, sub 1.0, gamma 0.2 | 0.9215 | **0.95521** | **+0.0019** |
| 9 | Ridge + 17 drifting features replaced by within-day ranks | 0.9243 | 0.94921 | **−0.0060** |
| 10 | Ridge + heavier balance: spw 24, mcw 16 | 0.9226 | 0.95475 | −0.0005 |
| 11 | EXPERIMENT: ridge trained on days 23–35 only | — | 0.95353 | −0.0017 |
| 12 | **50/50 rank-average: ridge + days-23–35 model** | — | **0.95629** | **+0.0011** |
| 13 | Equal average of models from start days 1, 12, 18, 23, 28 | — | 0.95396 | −0.0013 |
| 14 | 50/50 rank-average: ridge + days-18–35 model | — | 0.95612 | +0.0009 |
| 15 | **Ridge .5 + days-23 .25 + days-18 .25** | — | **0.95646** | +0.0002 |
| 16 | Same blend, 20 seeds per model | — | 0.95531 | −0.0012 |
| 17 | Best blend 0.8 + graph neural net 0.2 | — | 0.95547 | −0.0010 |
| 18 | Best blend + 25% within-day rank | — | 0.95476 | −0.0017 |
| 19 | Best blend − 25% within-day rank (sharpened days) | — | 0.95574 | −0.0007 |
| 20 | Graph neural net alone | — | 0.92745 | −0.0290 |
| 21 | 3-way blend + labelled-neighbour features | — | 0.95645 | −0.0000 |
| 22 | Ridge .6 + days-23 .2 + days-18 .2 | — | 0.95638 | −0.0001 |
| 23 | **Ridge .4 + days-23 .2 + days-18 .2 + days-12 .2** | — | **0.95672** | +0.0003 |
| 24 | Best blend 0.8 + neighbour-features-only model 0.2 | — | 0.94869 | −0.0080 |
| 25 | Best blend 0.7 + local-features-only model 0.3 | — | 0.95325 | −0.0035 |
| 26 | Ridge, spw24 model, days-23, days-18 at .25 each | — | 0.95626 | −0.0005 |
| 27 | Best blend 50/50 with LightGBM | — | 0.95347 | −0.0033 |
| 28 | Best blend rebuilt at 40 trees per model | — | 0.95244 | −0.0043 |
| 29 | Best blend 0.7 + ExtraTrees 0.3 | — | 0.95275 | −0.0040 |
| 30 | Ridge .4 + days-6/12/18/23 at .15 | — | 0.95608 | −0.0006 |
| 31 | 4-way with older d10 settings for the recent windows | — | 0.95546 | −0.0013 |
| 32 | Ridge .34 + days-12/18/23 at .22 | — | 0.95671 | −0.0000 |
| 33 | CatBoost alone | — | 0.94800 | −0.0087 |
| 34 | Random forest alone | — | 0.92618 | −0.0305 |
| 35 | Best blend 0.75 + CatBoost 0.25 | — | 0.95667 | −0.0001 |
| 36 | TabPFN alone, 1 bag | — | 0.96026 | +0.0035 |
| 37 | TabPFN 1 bag 0.5 + xgboost window blend 0.5 | — | 0.96149 | +0.0048 |
| 38 | TabPFN 3 bags 0.5 + xgboost window blend 0.5 | — | 0.96301 | +0.0063 |
| 39 | TabPFN 3 bags 0.65 + xgboost window blend 0.35 | — | 0.96389 | +0.0072 |
| 40 | **TabPFN all-days .40 + TabPFN days-18+ context .25 + xgboost .35** | — | **0.96529** | +0.0086 |
| 41 | TabPFN all-days .30 + days-18+ (3 bags) .35 + xgboost .35 | — | 0.96519 | +0.0085 |
| 42 | TabPFN all-days 5 bags .40 + days-18+ 3 bags .25 + xgboost .35 | — | 0.96506 | +0.0083 |
| 43 | **TabPFN all-days 5 bags .45 + days-18+ 3 bags .30 + xgboost .25** | — | **0.96570** | +0.0090 |
| 44 | TabPFN all 6 bags .40 + days-18+ .20 + days-23+ .20 + xgboost .20 | — | 0.96552 | +0.0088 |
| 45 | TabPFN all 6 bags .50 + days-18+ .35 + xgboost .15 | — | 0.96647 | +0.0098 |
| 46 | TabPFN only: all-days 6 bags .55 + days-18+ 3 bags .45 | — | 0.96692 | +0.0102 |
| 47 | TabPFN only, with one 12,000-row-context bag in the all-days part | — | 0.96703 | +0.0103 |
| 48 | TabPFN only, days-18+ share .55 | — | 0.96719 | +0.0105 |
| 49 | TabPFN only, days-18+ share .65 | — | 0.96732 | +0.0106 |
| 50 | Same, with a 12,000-row-context days-18+ bag in the recent share | — | 0.96805 | +0.0128 |
| 51 | **Recent share .75 (8k .40 + 12k .35)** | — | **0.96825** | +0.0130 |
| 52 | Recent share .75 split 8k / 12k / full-context .25 each | — | 0.96820 | +0.0130 |
| 53 | Recent share .80, 12k + full context only (no 8k bags) | — | 0.96775 | +0.0125 |
| 54 | Best blend, 12k recent slot averaged over 2 bags | — | 0.96797 | +0.0128 |
| 55 | Best blend, 12k recent slot = bag0 + 4-estimator bag | — | 0.96816 | +0.0130 |
| 56 | Best blend with a days-12+ TabPFN bag in the all-days share | — | 0.96800 | +0.0128 |
| 57 | Best blend .85 + CatBoost (days 18+) .15 | — | 0.96712 | +0.0119 |
| 58 | Best blend .85 + CatBoost (all days) .15 | — | 0.96708 | +0.0119 |
| 59 | Best TabPFN blend .80 + TabICLv2 (days-18+, full context) .20 | — | 0.96799 | +0.0128 |
| 60 | Best TabPFN blend .65 + TabICLv2 days-18+ full .175 + all-days full .175 | — | 0.96629 | +0.0111 |
| 61 | Best TabPFN blend .75 + three TabPFN v3 checkpoints (days-18+ 12k) .25 | — | 0.96701 | +0.0118 |
| 62 | Best blend, days 43–49 pooled rank shifted −0.05 | — | 0.96714 | +0.0119 |
| 63 | Best blend, days 43–49 pooled rank shifted +0.05 | — | 0.95519 | −0.0000 |
| 64 | Best blend, days 43–49 pooled rank shifted −0.02 | — | 0.96805 | +0.0128 |
| 65 | Best-blend recipe rebuilt with 8 TabPFN estimators per bag | — | 0.96694 | +0.0117 |
| 66 | Best-blend recipe rebuilt without the six time-marker features | — | 0.96402 | +0.0088 |
| 67 | Best TabPFN blend .75 + NVIDIA Kumo Tabular .25 | — | 0.97162 | +0.0164 |
| 68 | Best TabPFN blend .60 + Kumo (22 runs) .40 | — | 0.97280 | +0.0176 |
| 69 | **Best TabPFN blend .45 + Kumo (22 runs) .55** | — | **0.97357** | **+0.0184** |
| 70 | Best TabPFN blend .67 + Kumo (22 runs) .33 | — | 0.97229 | +0.0171 |

Of the first eight submissions, four improved the score. All four were changes to
model *settings*. Every change to the feature set or the model family failed.

From row 50 on, Δ is against the xgboost model of record, 0.95521. That flatters
every row that lost to the TabPFN blend: row 63 reads −0.0000 and cost 0.0131.
Against the best TabPFN blend (0.96825) rows 59–66 are −0.0003, −0.0020, −0.0012,
−0.0011, −0.0131, −0.0002, −0.0013 and −0.0042, and row 67 is **+0.0034**. Rows
59–67 are the last nine rows of `reports/experiments.csv`, which does not hold
rows 4–6.

The transfer ratio is shrinking as the model gets stronger: +0.0114→+0.0061,
+0.0090→+0.0052, +0.0115→+0.0019. Offline settings gains are still directionally
reliable but worth progressively less.

---

## 3. What won, and why it works

Only 11.67% of labelled transactions are illicit — 3,644 against 27,591. Left
alone the model can score well by leaning toward "clean", so it under-attends to
the rare cases. `scale_pos_weight 7.57` counts each illicit row as worth 7.57
licit ones; 7.57 is 27,591 ÷ 3,644, the ratio that makes both classes weigh the
same. It is not a tuned number.

That weighting has a side effect: tiny clusters of illicit transactions start to
look far more important than they are, and the trees begin carving rules around
two or three rows. `min_child_weight 8` sets a floor on how much evidence a
group must carry before a tree may split it, which stops that.

**Neither works alone.**

| | Offline Δ |
|---|---|
| `min_child_weight 8` alone | **−0.0056** |
| `scale_pos_weight 7.57` alone | +0.0046 |
| Both together | **+0.0090** |

This repeats a pattern from the earlier tuning round, where depth and colsample
gave +0.0035 alone and +0.0114 once the faster learning rate joined them.
One-setting-at-a-time sweeps find nothing here and would have concluded that
tuning does not help.

---

## 4. The central problem: offline validation does not predict the leaderboard

Submissions 4 and 5 both beat the standing model offline and both lost ground on
the leaderboard, one of them by 0.054. That triggered a full audit of the
validation method. Findings below are measured.

### 4.1 A regime break at day 23

Hold the days being predicted fixed at 30–35 and vary only the training history:

| Training days | Rows | AUC |
|---|---|---|
| 1–21 | 19,530 | 0.8878 |
| 1–22 | 21,293 | 0.8739 |
| **1–23** | 20,717 | **0.9938** |
| 1–24 | 23,606 | 0.9964 |
| **day 24 alone** | **1,126** | **0.9922** |
| 1–15 | 15,514 | 0.8472 |

One day of data — 1,126 rows — outperforms twenty-two days and 21,293 rows. It
is not volume, it is which regime the training set has seen. Days 1–22 belong to
one regime and days 23 onward to another.

**This invalidates the validation folds.** The competition asks for 14 days of
prediction, so a fold that mimics the task must hide 14 of the 35 labelled days,
forcing training to stop at day 21. The break is at day 23. The rule that makes
a fold realistic is the same rule that puts it on the wrong side of the break,
and they miss each other by two days.

The submitted model trains on days 1–35 and is on the correct side. So the folds
were grading a handicapped model that is never submitted.

### 4.2 The alternative is worse

Move a fold past the break and it goes blind. Eight very different models —
depth 4 through 10, all 165 features, 93 features, 72 features, a random 120 —
scored on train 1–24 → validate 25–35:

    0.9959 to 0.9971, total spread 0.0012

A randomly chosen feature subset comes out on top. There is no headroom left.
Windows with room to discriminate are on the wrong side of the break; the window
on the right side has none.

### 4.3 No window choice fixes it

Four submitted models with known leaderboard scores, scored under five
protocols. A usable protocol must reproduce the Kaggle order.

| Protocol | Rank correlation with Kaggle |
|---|---|
| 3-fold harness (current) | **+0.40** |
| primary fold only | +0.40 |
| regime-shifted, train 1–22 | 0.00 |
| regime-matched, train 1–29 | 0.00 |
| early, train 1–13 | −0.20 |

All five rank the known leaderboard loser **first**. The existing harness is the
best of them, and only because it gets the three settings comparisons right.

### 4.4 Reweighting toward the test distribution is impossible

A classifier tells a training row from a test row with **AUC 1.0000** on
features alone. No training row resembles a test row, so there is nothing to
reweight toward. `feat_101`, `feat_100` and `feat_103` are the giveaway — over
99% of test rows fall outside the training range of each. Six features separate
the two periods at above 0.90 on their own; they behave as time markers.

### 4.5 The conclusion

Three categories of change, three different verdicts:

| Category | Transfers? | Record |
|---|---|---|
| Settings, seeds, averaging | **Yes**, at about half the offline size | 3/3 |
| Feature set | **Inverts** | 0/2 |
| Model family | **Discounted to zero** | 0/1 |

Settings gains have now transferred three times running at roughly half their
offline magnitude: +0.0114 → +0.0061, +0.0090 → +0.0052. That halving is the
only forecasting rule this project has produced that has held up.

---

## 5. Data inspection

No cleaning was required.

| Check | Result |
|---|---|
| Missing values in the 165 features | 0, both files |
| Missing values elsewhere | 110,537, all in `label` — the unlabelled rows |
| Duplicate transaction IDs | none; no overlap between train and test |
| Duplicate feature rows | 31 in train (0.02%) |
| Constant features | 0 |
| Labelled rows | 31,235 — 27,591 licit, 3,644 illicit (11.67%) |

**Scaling, centring, log transforms and outlier clipping were all unnecessary.**
A decision tree only asks whether a feature is above some threshold, so any
transform preserving order leaves the model unchanged. Features range from −13
to +288 and 45 have skew above 10; none of it matters for a tree. It matters
enormously for a neural net — see §6.

**Redundancy.** 36 feature pairs correlate above 0.99, and six pairs are exactly
identical: `feat_5`=`feat_14`, `feat_23`=`feat_29`, `feat_24`=`feat_30`,
`feat_59`=`feat_65`, `feat_60`=`feat_66`, `feat_61`=`feat_67`.

**Signal.** The strongest single feature is `feat_53` at 0.857 AUC on its own.
54 of the 165 are near-useless alone. A transaction's own features carry more
signal than its neighbours' summaries — mean solo AUC 0.648 for `feat_1`–`feat_93`
against 0.600 for `feat_94`–`feat_165`.

**Graph.** 203,769 nodes and 46,668 of them have no feature row. Median degree
2, 34.5% of nodes have exactly one link. Of the 175,918 edges with both ends
known, **zero cross a day boundary** — the graph is 49 disconnected daily
components, so label propagation across days is impossible.

---

## 6. Model families tested

Seven learners over the same folds and seeds:

| Model | Offline AUC |
|---|---|
| **Neural net (MLP)** | **0.9289** |
| XGBoost | 0.8959 |
| LightGBM | 0.8773 |
| sklearn hist-boost | 0.8696 |
| Random forest | 0.8487 |
| Extra trees | 0.8481 |
| Logistic regression | 0.8456 |

Tuned to `(256,128)` with `alpha=1e-2` and a quantile transform, the MLP reached
0.9400, and 0.9434 seed-averaged — **+0.047 over XGBoost**, the largest offline
effect measured in this project.

Preprocessing was decisive here and nowhere else: the same network scores 0.9400
with a quantile transform and **0.8748** with a plain standard scaler.

It did not convert. A 50/50 blend scored 0.94800 against a standing 0.94812.
Since rank-averaging two models of equal skill normally gains a little, a blend
landing exactly on the stronger one implies the MLP is the weaker model on days
36–49. Two warning signs were visible beforehand: the MLP loses to XGBoost on
windows past the day-23 break (0.9789 against 0.9953), and the two models
correlate at only **0.59 on the real test rows** against 0.85 on validation days.

---

## 7. Ruled out

| Direction | Result |
|---|---|
| Hand-built graph features (degree, two-hop, neighbour aggregates) | No gain; cost 0.054 on the leaderboard |
| The 110,537 unlabelled rows — PCA features, pseudo-labelling in 10 variants | Best outcome a tie, worst cost 0.093 |
| Dropping distribution-shifted features | +0.0079 offline, −0.0083 on the leaderboard |
| LightGBM, alone or blended | 0.8792 against 0.9030; every blend weight worse |
| Neural net, alone or blended | +0.047 offline, −0.0001 on the leaderboard |
| Recomputing neighbour averages from the edge list | `feat_94`–`feat_165` already contain these |
| Training on recent days only | Flat past the break; the apparent gain was one fold |

Pseudo-labelling failed because the guesses are the model's own beliefs fed back
as fact, so it mislabels confidently exactly where it is already wrong. Damage
scaled monotonically with the guessed labels' total influence, optimum at zero.

The graph-feature failure has a precise cause worth recording. Two engineered
features counted "neighbours with no feature row". Because `train.csv` holds
every node for days 1–35 while `test.csv` holds only labelled nodes for days
36–49, that count is **exactly 0 for every training row and non-zero for 53% of the
test rows** (the means below are over all test rows):

```
g_unknown_neighbours   train mean 0.0000   test mean 1.5816
g_unknown_fraction     train mean 0.0000   test mean 0.3910
```

The model learned nothing from a constant and then met live values it had never
seen. `validation.assert_no_shift` was written to catch exactly this and must
run on any engineered feature before a submission.

---

## 8. Measurement discipline

A single training run varies by **±0.013 AUC** from its own randomness alone,
which is wider than the gap between first and last place. Every number in this
report is the mean of 3 time folds × 5 or 10 seeds, with a standard error.
Three apparent improvements in this project turned out to be noise.

Two independent noise sources:

| Source | Size | Mitigation |
|---|---|---|
| The model's own randomness | ±0.013 per run | Average over seeds |
| The leaderboard's 30% sample | ±0.011 per score | None — only respect it |

The public leaderboard scores roughly 4,598 rows holding about 271 illicit
transactions. A single score there has a standard deviation of 0.0106, against a
top-six spread of about 0.0022. A true gap of 0.0013 reverses about a third of
the time. Only offline settings gains above ~0.005 justify a submission.

---

## 10. Branch `model/settings-sweep-v2` — further iteration

Not submitted. Recorded here so none of it is repeated.

### 10.1 What worked: a settings ridge, +0.0115 offline → +0.0019 on the board

A 24-configuration random search over whole combinations, rather than one axis
at a time. Only 3 of 24 beat the current model, none by more than 0.0025 — but
all three shared `subsample=1.0`, `colsample_bytree=0.7`, `reg_lambda=1.0`. That
ridge was then tested directly rather than taking the single best sample.

| Config | Seeds 0–9 | Seeds 10–19 (unseen by the search) |
|---|---|---|
| **d12, cs 0.7, sub 1.0, mcw 8, gamma 0.2** | **+0.0110** (p=0.0001) | **+0.0115** (p<0.0001) |
| d10, cs 0.7, sub 1.0, mcw 8 | +0.0086 | +0.0079 |
| d10, cs 0.7, sub 1.0, mcw 5, gamma 0.2 | +0.0059 | — |
| d12, cs 0.7, sub 1.0, mcw 5, gamma 0.2 | +0.0040 | — |

**49 wins in 60 paired fold-and-seed comparisons.** Crucially there is no
shrinkage between the two seed halves. The earlier `spw + mcw` result fell from
+0.0112 to +0.0062 on re-measurement, which is what best-of-fifteen selection
looks like. This one held, which is the main reason to believe it.

Code in `src/model_ridge.py`, search in `src/tune_v2.py`, all 24 results in
`reports/tuning_v2.csv`.

### 10.2 Seed count: no evidence for widening it

Rank-average of the ridge model over 1 to 20 seeds, on the standard folds:

| Seeds | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| Mean AUC | 0.9231 | 0.9236 | 0.9218 | 0.9222 | 0.9226 |

Flat to within 0.0018. No offline case for going past 5. Note the harness also
measured seed averaging at +0.0000 when it delivered +0.0030 on the leaderboard,
so read this as "cannot see a difference" rather than "there is none".

### 10.3 What failed

**Training objective and evaluation metric** — every variant lost ground against
the ridge model's 0.9208:

| Variant | Offline | Δ |
|---|---|---|
| `binary:logistic` / `auc` (current) | 0.9208 | — |
| `eval_metric=aucpr` | 0.9173 | −0.0035 |
| `max_delta_step=1` | 0.9046 | −0.0161 |
| `eval_metric=logloss` | 0.9011 | −0.0196 |
| `objective=rank:pairwise` | ~0.749 | **−0.17** |

The ranking objective is worth a note. AUC scores ordering, so training directly
on ordering looks like the obviously right move, and it is dramatically worse.
It also needs `XGBRanker` with explicit groups — driven through `XGBClassifier`
it silently returns constant predictions and a flat 0.5000.

**Averaging different configurations instead of different seeds.** The reasoning
was that five seeds of one config all make the same kind of mistake, whereas
five different configs make different mistakes:

| Ensemble | Offline |
|---|---|
| Ridge alone, 5 seeds | **0.9218** |
| 3 configs × 5 seeds = 15 models | 0.9200 |
| 5 configs × 5 seeds = 25 models | 0.9184 |

Worse, and monotonically so. The other configs are genuinely weaker, and
averaging drags the best one down toward them. Diversity does not pay when it is
bought with quality.

**CatBoost**, a third boosting library whose "ordered boosting" is designed to
reduce exactly the quiet overfitting boosting is prone to:

| Depth | 4 | 5 | 6 | 8 |
|---|---|---|---|---|
| Offline AUC | **0.9054** | 0.8963 | 0.8905 | 0.8653 |

Best at depth 4 and still 0.015 behind XGBoost.

### 10.4 The model-family tally is now seven

XGBoost beats LightGBM, CatBoost, scikit-learn's hist-boost, random forest,
extra trees, logistic regression and a tuned neural net. The neural net is the
only one that beat it offline, and it went flat on the leaderboard.

**Stop testing model families.** Seven have been tried, the search space is
exhausted, and the one that looked best offline is the one that cost a
submission to disprove.

---

## 9. Where this could still go

Remaining directions, all in the only category that has transferred:

Both directions listed here have now been run — see §10. The settings sweep
found a real ridge worth **+0.0115** offline, which on the halving rule projects
to roughly **+0.006** on the leaderboard. Widening the seed average found
nothing.

That candidate was submitted on 2026-09-17 and scored **0.95521**, +0.0019. It
is now the model of record. The projection was +0.006, so the forecast was three
times the delivered gain.

Beyond it the settings space around the current model has been searched twice,
seven model families have been tested and beaten, and the remaining headroom
looks thin. Second place moved from 0.94890 to 0.95357 over the same period, so
the field is closing. The top of the
leaderboard spans 0.0022, so that is enough to matter, but it is tuning rather
than a breakthrough.

Anything touching the feature set or the model family has a 0-for-3 record and
no offline measurement can screen it first. Such a change is a submission spent
to buy information, and should be described that way.

---

## 11. 2026-09-26 — three submissions, none improved

Standing at the start: 4th, 0.95521 against a leader at 0.95741.

**Drifting features as within-day ranks (0.94921, −0.0060).** 17 features whose
value alone separates train from test at AUC > 0.75 were replaced by their
percentile rank among the same day's labelled rows. Offline this was the most
convincing feature result so far: +0.0085 (seeds 5–14), +0.0118 (seeds 15–24,
27/30 wins), a plateau over thresholds 0.65–0.80, and positive even on the
saturated post-cliff windows (+0.0004, 9/10 wins). The leaderboard inverted it
anyway. Likely cause: pooled AUC ranks across days, and ranking within a day
throws away the between-day level differences that the raw values carry.
Feature-set changes are now **0 for 3**.

**Heavier class balance, spw 24 + mcw 16 (0.95475, −0.0005).** A 3×3 grid of
spw {12,16,24} × mcw {12,16,24} was positive in every cell; the top cell
replicated without shrinkage (+0.0073 → +0.0071 on unseen seeds). It landed flat.
This is the first settings change to not transfer. Five xgboost variants now sit
in 0.9547–0.9552 on the public board: the settings direction looks exhausted.

**Days 23–35 only (0.95353, −0.0017).** An experiment. It uses a third of the
rows and still comes within 0.0017 — while ranking the test rows differently from
the full model (Spearman 0.896). Two near-equal, different models are the one
blending setup not yet tried here; earlier blends paired a strong model with a
weaker or near-identical one.

Also measured, not submitted: labelled-neighbour counts (a node counts as
labelled if it has a label in train or a row in test.csv — symmetric by
construction). +0.0025 on seeds 0–4, +0.0004 on seeds 5–14. Noise.

---

## 12. 2026-09-27 — window blending, first blend to gain

A new leader appeared overnight at 0.96362 (+0.006 over the previous top).
Checked for a leak first: transaction id (AUC 0.56, 0.50 post-cliff) and
within-day row position (0.55) carry at most weak signal. Not the explanation.

| Blend | Leaderboard |
|---|---|
| ridge 50 / days-23–35 50 | **0.95629** (best) |
| ridge 50 / days-18–35 50 | 0.95612 |
| five windows equal (1, 12, 18, 23, 28) | 0.95396 |

Two different recent windows each add ~+0.001 when paired 50/50 with the full
model, so it is probably real rather than sampling luck. It fails when the full
model is diluted to 20% and a 6,312-row window (days 28–35, Spearman 0.80 with
the full model) joins. Models: `src/model_recent.py`, `src/model_windows.py`;
blending: `src/blend.py`.

---

## 13. 2026-09-28 — the leaderboard's noise floor, measured directly

The 3-way blend scored 0.95646. The same blend with 20 seeds per model instead
of 5 scored 0.95531, although the two files agree at Spearman **0.998**. So two
near-identical submissions differ by **0.0012** on the public board. That is as
large as every blend gain recorded since 2026-09-27 (+0.0009 to +0.0013). Those
gains cannot be told apart from sampling noise, and choosing between them on the
public score only fits the 30% public sample.

A local-features-only model (93 features) was checked as a blend partner
offline. The blend lost on four of five windows (up to −0.008 pre-cliff), gained
+0.0005 on one. Not submitted; the third slot was left unused.

---

## 14. Graph neural network (`src/gnn.py`, `src/model_gnn.py`)

Two layers of directed message passing (incoming and outgoing neighbours
aggregated separately), quantile-scaled inputs. **Features only on labelled
nodes, on both sides**: train's unlabelled rows are blanked to match the
featureless unlabelled nodes of days 36-49, with a has-features flag and log
degree. Full-batch training on a day subgraph (no edge crosses a day, so a range
of days is an exact subgraph).

| Window | xgboost | GNN | blend 20% GNN |
|---|---|---|---|
| train ≤21 → 22–35 | 0.9187 | 0.888–0.917 (noisy) | 0.9145 |
| train ≤24 → 25–35 | 0.9965 | 0.9879 | 0.9973 |
| train ≤29 → 30–35 | 0.9972 | 0.9928 | 0.9977 |

Rank correlation with xgboost 0.72–0.76 offline, 0.69 on test. Submitted at 20%
on top of the best blend: **0.95547**, no gain. Same pattern as the MLP.

## 15. Where the test period differs: days 43–49

The model scores days 43–49 almost entirely clean: under 1% of rows above 0.5,
against 6–14% on days 37–42. If those days hold illicit transactions of a kind
the model has not seen, pooled AUC loses every pair where they sit below a
clean day-36–42 row. `src/dayrank.py` mixes each score with its within-day rank.
Offline (days 1–35) that only loses: −0.003 at 25%, −0.009 at 50%. Queued as a
leaderboard experiment.

**Result (2026-09-29).** Both directions lost: +25% within-day rank 0.95476,
−25% 0.95574. The blend's between-day balance is already near its optimum on
the test days, and the model's low scores on days 43–49 look right. Lead closed.
The GNN alone scored **0.92745**, too weak to pull a blend up. GNN route closed.

---

## 16. 2026-09-30

Two teams now sit above 0.962 (0.96362, 0.96258); everyone else is below 0.959.

- **Labelled-neighbour features** (`src/model_nlab.py`) added to all three window
  models: 0.95645, identical to the base. Closed.
- Offline, not submitted: **unlabelled training rows as clean negatives** −0.03
  to −0.16 pre-cliff, ±0.001 post-cliff; **day number as a feature** +0.009 to
  −0.005 across windows. **Test row order** correlates 0.07 with predictions.
  Not a leak.
- Near-best variants: 4-way (ridge .4, days-12/18/23 at .2 each) **0.95672**,
  new best; ridge .6 variant 0.95638. Both within the ~0.001 noise band.

---

## 17. 2026-10-01 — feature-subset partners both lose

Offline, a neighbour-only model (feat_94–165) beat the full model across the
regime cliff (0.9453 vs 0.9163 on train ≤15 → 16–29) and lost within a regime
(0.9299 vs 0.9972 on 30–35), with rank correlation 0.43–0.62. Blended at 20%:
**0.94869**, −0.008. The test days behave like the post-cliff regime.

The mirror, a local-only model (feat_1–93) that pre-cliff windows disliked, at
30%: **0.95325**, −0.0035. So "the leaderboard inverts the pre-cliff windows"
is not a usable rule either; the full feature set is simply best. Both subset
partners are closed. Code: `src/model_neigh.py`, `src/model_local.py`.

### Post-cliff windows as the judge for blend partners

Five post-cliff windows (train ≤k, score the next 6 days, k = 23–31, 2 seeds):

| Partner | Alone vs xgboost | 30% blend | 50% blend |
|---|---|---|---|
| LightGBM | +0.0001 | +0.0004 (10/10) | +0.0005 (10/10) |
| ExtraTrees | −0.0020 | +0.0010 (10/10) | +0.0009 (9/10) |
| RandomForest | −0.0062 | −0.0002 | −0.0012 |

Both families were rejected earlier on the pre-cliff windows. Queued:
`blend4_et30`, `blend4_lgbm50`, `blend4_lgbm25_et25` (`src/model_families.py`).
On test, ExtraTrees correlates only 0.685 with the best blend and LightGBM 0.933.

**Result (2026-10-02).** LightGBM 50%: 0.95347; ExtraTrees 30%: 0.95275. The
post-cliff windows misjudge model families as badly as the pre-cliff ones did.
Fewer trees (40 instead of 79) also lost (0.95244), as more trees did before:
79 trees at learning rate 0.1 is near the optimum.

Tally of everything that has gained on the leaderboard: settings (spw/mcw, tree
shape), seed averaging, and blending the same xgboost model trained on
different recent windows. Every feature change, every other model family
(MLP, GNN, LightGBM, ExtraTrees, neighbour-only, local-only xgboost) and every
post-processing step has lost.

**2026-10-03.** Three more window-blend variants: 0.95608, 0.95546, 0.95671.
None beat 0.95672. The window-blend direction has plateaued.

---

## 18. 2026-10-05 — research, no submissions

Five teams now above 0.96 (leader 0.96805). Several jumped from ~0.957 to
0.961+ within the same two days, which suggests a shared hint rather than five
independent discoveries.

- **Competition Discussion tab:** not reachable. The page needs a login, and the
  Kaggle API returns 403 Forbidden for discussions with this token. No public
  notebooks exist for the competition.
- **Literature on the same day-based split** (train up to step 34, test
  35–49): XGBoost is the strongest reported model (PR-AUC 0.807 vs random
  forest 0.786). Every published model collapses after the step-43 dark-market
  shutdown, and no paper reports a fix. A feature-gated temporal GNN does not
  beat random forest on F1 (0.774 vs 0.796). Nothing found beats the approach
  already used here.

---

## 19. 2026-10-06 — other model families, judged only on the leaderboard

| Model (all features, all days, 5 seeds) | Leaderboard |
|---|---|
| xgboost (model of record) | 0.95521 |
| CatBoost (600 iters, depth 6, balanced) | 0.94800 |
| Random forest (1000 trees) | 0.92618 |
| Best blend 0.75 + CatBoost 0.25 | 0.95667 (tie with best) |

TabPFN (pretrained tabular foundation model) installed but needs a licence
token from priorlabs.ai tied to the user's account before weights download.
Code: `src/model_alt.py`.

---

## 20. TabPFN — the first model family to beat xgboost

TabPFN is a pretrained tabular foundation model: it reads a sample of labelled
rows as context and predicts new rows in one forward pass, with no training.
`src/model_tabpfn.py` bags it on CPU: each bag keeps all 3,644 illicit rows plus
4,356 sampled licit rows (8,000-row context), and bags are rank-averaged.

| Submission | Leaderboard |
|---|---|
| TabPFN alone, 1 bag | 0.96026 |
| TabPFN 1 bag 0.5 + xgboost window blend 0.5 | 0.96149 |
| TabPFN 3 bags 0.5 + xgboost window blend 0.5 | 0.96301 |
| TabPFN 3 bags 0.65 + xgboost window blend 0.35 | **0.96389** |

On 3,000 rows from days 20–29 scored on days 30–35 it beat xgboost trained on
the same rows (0.9978 vs 0.9955). TabPFN and the xgboost blend correlate 0.85 on
test. Bagging and a heavier TabPFN weight both helped. Cost: ~18 minutes per bag
on 4 CPU cores. Needs `TABPFN_TOKEN` (licence at priorlabs.ai).

Recent-days context helps TabPFN as it helped xgboost: a bag whose context is
drawn from days 18–35 only (all 2,265 illicit rows there plus sampled licit),
mixed in at 25%, gave **0.96529**.

More bags stopped helping after three (5 all-days bags: 0.96506, a tie). A larger
TabPFN share keeps helping: .50 → .65 → .75 gave 0.96301 → 0.96389 → 0.96570.

TabPFN share, end to end: .50 → .65 → .75 → .85 → 1.0 gave 0.96301 → 0.96389 →
0.96570 → 0.96647 → **0.96692**. With enough bags (6 all-days + 3 days-18+),
TabPFN alone beats every mix with xgboost.

A larger context helps most in the recent-days TabPFN: one 12,000-row-context
days-18+ bag, given .25 of the .65 recent share, moved 0.96732 → **0.96805**.
Days 18–35 hold 14,380 labelled rows, so the next step is to use all of them.

Full 14,380-row context for days 18–35 adds nothing over 12,000 (Spearman 0.986
between them; 0.96820, a tie). Dropping the 8,000-row recent bags lost 0.0005:
the mix of context sizes is itself a source of useful diversity.

Note: the container is reclaimed a few minutes after the session goes idle, so
"overnight" TabPFN runs only progress while a session is active.

Re-weighting and adding more bags of the same TabPFN settings has plateaued at
~0.968 (0.96797–0.96825, all within public noise). A 4-estimator bag is no
better than a 2-estimator one.

A days-12+ window adds no new information either (Spearman 0.965–0.970 with the
existing bags; 0.96800). Every TabPFN variant on 2026-10-12/13 landed between
0.96775 and 0.96825.

---

## 21. CatBoost as a partner to TabPFN — loses

CatBoost ranks test rows differently from the best TabPFN blend (Spearman
0.73–0.75), but at a 15% share it cost 0.0011 either way (all days 0.96708,
days 18+ 0.96712). Like xgboost and LightGBM, it is too weak next to TabPFN to
add anything. The model of record stays the TabPFN-only blend at **0.96825**
(`submissions/blend_tabpfn_recent75_12k.csv`). Superseded on 2026-10-23, see §27.

---

## 22. 2026-10-21 — a GPU machine, and the old bags were already v3.5

Work moved to a Windows 11 PC with an NVIDIA RTX 3060 Ti (8 GB). TabPFN bags that
took ~18 minutes on 4 CPU cores (§20) now take a few minutes.

| Item | Setting |
|---|---|
| Python | 3.11 |
| `.venv` | `--system-site-packages`, system torch 2.5.1+cu124, packages installed unpinned |
| TabPFN | `tabpfn` 9.2.0, default weights TabPFN v3.5 |
| TabICLv2 | `tabicl` 2.2.0 |
| `.venv-fm` (gitignored) | torch 2.11.0+cu128, `structured-data-models` 0.2.1 (Kumo Tabular), `huggingface_hub` 2.2.0 |

The pinned `requirements.txt` does not install on Python 3.11: `numpy==2.5.3`
needs a newer Python. Everything from 2026-10-21 on ran in the unpinned
environments above.

**Weights and licences.** TabPFN v3, v2.6 and v2.5 weights need a one-time
licence acceptance at `ux.priorlabs.ai/account/licenses`. v3.5 worked with the
existing `TABPFN_TOKEN`. Kumo weights come from Hugging Face, `nvidia/Kumo-Tabular`
(revision v1.0.1, `large`, ~855 MB), under the OpenMDW 1.1 licence, not gated.

**Code.** `src/model_tabpfn.py` now takes its settings from environment variables:

| Variable | Meaning |
|---|---|
| `TABPFN_CONTEXT` | Rows per bag, default 8000. 99999 means all rows from the first day |
| `TABPFN_ESTIMATORS` | Internal estimators per bag, default 2 |
| `TABPFN_CHUNK` | Test rows per predict call, default 2000 |
| `TABPFN_VERSION` | `v2_5`, `v2_6`, `v3`, `v3_5` or `tabicl` (TabICLv2). Unset keeps the package default and the old file names |
| `TABPFN_CKPT` | A specific checkpoint file of that version |
| `TABPFN_CACHED=1` | `fit_with_cache`. Runs out of memory at 14k context or more on 8 GB |
| `TABPFN_DROP` | Comma-separated features to leave out |
| `TABPFN_CLIP=1` | Clip test features to the bag's context range |
| `TABICL_BATCH` | TabICL batch size, default 2 |
| `TABPFN_DEVICE` | Force a device instead of auto-detecting |

Output files are named `tabpfn{version}_{tag}_bags{a}-{b}.csv`. `src/blockshift.py`
(§24) and `src/fm/` (§27) are new.

### 22.1 The rebuild check

A rebuild of the days-18+ 8k 3-bag set with v3.5 on the GPU, compared with the
old bags:

| Comparison | Spearman on test |
|---|---|
| GPU v3.5 rebuild vs the old days-18+ 8k bags | 0.997 |
| Same bag, GPU vs CPU | 0.989 |

Two consequences:

- **The earlier runs were already TabPFN v3.5**, the package default. There was no
  free gain from upgrading. What had never been run was every other set of
  weights, which is §23.
- **0.989 is the floor for "same recipe".** One bag run on two devices already
  differs by that much, so a variant that correlates 0.99 with its baseline is
  indistinguishable from device noise.

### 22.2 Run times on the RTX 3060 Ti

| Run | Time |
|---|---|
| TabPFN v3.5, 8k bag, 2 estimators | ~3.5 min |
| TabPFN v3.5, 12k bag | ~5–8 min |
| TabPFN, 8 estimators | ~4× the 2-estimator time |
| TabPFN v3.5, full 31,235-row context (non-cached, chunk 1000) | 26 min |
| TabICLv2, full context | 1–3 min |
| Kumo Tabular large, 8k context | ~2 min |
| Kumo Tabular large, 31k context | 18 min |

---

## 23. 2026-10-21 — other foundation models as TabPFN partners

CatBoost lost as a partner to TabPFN (§21), and xgboost's share in the best mix
had already shrunk to zero (§20). The open question was whether another
pretrained tabular model, or other TabPFN weights, could do what they could not. Three submissions, each added to the best TabPFN blend:

| # | Partner | Share | Spearman vs best | Public score | vs 0.96825 |
|---|---|---|---|---|---|
| 59 | TabICLv2, days-18+ full context | .20 | 0.934 | 0.96799 | −0.0003 |
| 60 | TabICLv2, days-18+ full .175 + all-days full .175 | .35 | 0.92–0.93 | 0.96629 | −0.0020 |
| 61 | TabPFN v3: default, binary and ood checkpoints (days-18+ 12k) | .25 | 0.87–0.91 | 0.96701 | −0.0012 |

TabICLv2 at 20% is a tie inside the noise of §13. At 35% it costs 0.0020. The v3
checkpoints rank the test rows differently from the best blend (0.87–0.91) and
still lose, so lower correlation alone was not enough.

Two more candidates did not finish on 8 GB:

| Model | What happened |
|---|---|
| TabPFN v2.6, 12k context | 573 s per 2000-row chunk (memory thrashing). Killed |
| LimiX-2 (`stable-ai/LimiX-2`, 400M parameters, non-commercial licence) | No chunk finished in 20+ minutes with 8 or 4 pipelines. With 2 pipelines it took 8.5 minutes. Dropped |

---

## 24. 2026-10-22 — days 43–49: the block offset, optimum at zero

§15 found that the xgboost model scores days 43–49 almost entirely clean, and
tried to fix the balance between those days and days 36–42 with within-day
ranks. Both directions lost. This is a cleaner version of the same test, on the
best TabPFN blend.
`src/blockshift.py` adds one constant to the pooled percentile rank of every test
row from day 43 on. The order inside each block does not change; only where the
block sits against days 36–42.

| # | Shift on days 43–49 | Public score | vs 0.96825 |
|---|---|---|---|
| 62 | −0.05 | 0.96714 | −0.0011 |
| 64 | −0.02 | 0.96805 | −0.0002 |
| 51 | 0 (best blend) | 0.96825 | — |
| 63 | +0.05 | 0.95519 | −0.0131 |

The optimum is at about zero. Moving the block down costs little. Moving it up by
the same 0.05 costs about twelve times as much. That is consistent with §15: if
the low scores on days 43–49 are right, lifting those days puts clean rows above
illicit ones from days 36–42. The blend's balance between the two blocks needs no
correction. Post-processing has not gained anywhere: two within-day rank mixes
and three block offsets all landed at or below the model they were applied to.

---

## 25. 2026-10-23 — variants inside TabPFN

Everything below changes how TabPFN is run and keeps the model the same.

| Variant | Compared with | Spearman | Public score | vs 0.96825 |
|---|---|---|---|---|
| Clip test features to the context range (12k bag, GPU) | Same bag, unclipped | 0.992 | not submitted | — |
| Clip test features to the context range (8k bags) | Same bags, unclipped | 0.995 | not submitted | — |
| Full 31,235-row context, non-cached, chunk 1000 | Best blend | 0.971 | not submitted alone | — |
| 8 estimators per bag instead of 2, whole recipe rebuilt (#65) | Best blend | — | 0.96694 | −0.0013 |
| Six time-marker features left out, whole recipe rebuilt (#66) | Best blend | — | 0.96402 | −0.0042 |

**The time markers.** Six features separate days 18–35 from the test days at AUC
above 0.9: `feat_101` 0.998, `feat_103` 0.997, `feat_100` 0.996, `feat_136` 0.994,
`feat_139` 0.953 and `feat_137` 0.910. `feat_100`, `feat_101` and `feat_103` lie
outside the training range for over 99% of test rows (§4.4). Two ways to remove
the risk were tried.

- **Clip** the test values into each bag's context range. The ranking barely
  moves (0.992 and 0.995), so there is nothing to submit.
- **Drop** the six features. This costs 0.0042, the largest loss of the TabPFN
  variants in this section. TabPFN seems to get useful signal from these
  features even where the test values are out of range. xgboost did the same
  when 21 shifted features were dropped (row 5): +0.0079 offline, −0.0083 on the
  board.

**Estimators.** Eight estimators per bag take four times as long and score 0.0013
below the best blend. The earlier 4-estimator bag was no better than a 2-estimator
one either (§20). Internal ensembling is not where the gain is.

**Context size.** The full 31,235-row context fits on 8 GB without the cache (the
cached version runs out of memory at 14k or more) and takes 26 minutes. Its
Spearman with the best blend is 0.971. It was not submitted alone. The 14,380-row
full context for days 18+ had already tied the 12k bag (§20).

**Other windows.** Days-12+ (row 56, 0.96800) added nothing. A days-23+ bag in the
mix of row 44 scored 0.96552, under the neighbouring mixes without it (0.96570,
0.96647).

Every TabPFN-side variant since 2026-10-12 sits between 0.96402 and 0.96825, and
none beats the best blend.

---

## 26. Exact-match re-identification features — no gain

An idea checked offline only. For each row, count the earlier labelled rows that
share an exact value of a feature, and give the counts to xgboost.

| Window | Baseline | With exact-match counts | Δ |
|---|---|---|---|
| train ≤21 | 0.9139 | 0.9108 | −0.0031 |
| train ≤29 | 0.9971 | 0.9973 | +0.0002 |

No gain on either window, and a loss on the pre-break one. Not submitted. Feature
changes keep their record: nothing in that category has gained on the board.

---

## 27. 2026-10-23 — Kumo Tabular, the first partner that adds to TabPFN

Since 2026-10-12 every TabPFN variant, every bag mix and every partner (xgboost,
CatBoost, TabICLv2, TabPFN v3) landed at or below 0.96825. NVIDIA Kumo Tabular is
a different pretrained tabular model that also reads labelled rows as context.

### 27.1 What was run

`src/fm/run_kumo.py` (run from `src/fm` with `.venv-fm`) builds its context the
way the TabPFN bags do: every illicit row from the first day on, plus seeded
random licit rows up to the context size. Model size `large`, 8 estimators.
Usage: `python run_kumo.py <first_day> <context_rows> <seed> <out_csv> [device]`.

| Context | Rows | Runs |
|---|---|---|
| Days 18+ | 8,000 | seeds 0–4 |
| Days 18+, full | 14,380 | seed 0 |
| All days | 12,000 | seeds 0–4 |
| All days, full | 31,235 | seed 0 |

Twelve runs, about 2 minutes each at 8k and 18 minutes at 31k. The Kumo component
(`kumo_blend.csv`) gives the days-18+ runs .125 each (.75 in total) and the
all-days runs .04167 each (.25 in total). The final blend:

    python src/blend.py blend_r75_kumo25.csv blend_tabpfn_recent75_12k.csv:0.75 kumo_blend.csv:0.25

### 27.2 Result

| # | Model | Public score |
|---|---|---|
| 51 | Best TabPFN blend | 0.96825 |
| 67 | **Best TabPFN blend .75 + Kumo .25** | **0.97162** |

+0.0034 over the TabPFN-only blend and +0.0164 over the xgboost model of record.
It was 2nd when submitted and 3rd later that evening: 2nd is 0.97190 and the
leader 0.97446. The gap to 2nd is 0.0003, inside the 0.0012 noise floor of §13, so
the order of 2nd and 3rd means little. The gap to the leader is 0.0028.

The gain is about three times that noise floor. It is larger than any blend gain
in the xgboost era (+0.0009 to +0.0013) and than the whole spread of the TabPFN
plateau (0.96775 to 0.96825, 0.0005). It is still one submission on the 30%
public sample.

### 27.3 Why it worked

Kumo ranks the test rows differently from TabPFN, and its own runs agree with
each other.

| Rank correlation (Spearman, test) | Value |
|---|---|
| One Kumo seed vs another | 0.98 |
| Averaged Kumo vs best TabPFN blend | 0.863 |
| Single Kumo run vs best TabPFN blend | 0.83–0.91 |

The 0.98 matters. The 12-run average is a stable second opinion, so its
disagreement with TabPFN does not come from run-to-run randomness.

Correlation alone does not explain it. Every partner added to the TabPFN blend:

| Partner | Rank correlation with the TabPFN side | Result |
|---|---|---|
| xgboost window blend | 0.85 (§20, 3-bag TabPFN) | Lost: the mix with xgboost at .15 scored 0.96647, TabPFN only 0.96692 (rows 45, 46) |
| CatBoost | 0.73–0.75 | Lost 0.0011 at .15 |
| TabICLv2 | 0.92–0.93 | −0.0003 at .20, −0.0020 at .35 |
| TabPFN v3 checkpoints | 0.87–0.91 | −0.0012 at .25 |
| **Kumo Tabular** | **0.86** | **+0.0034 at .25** |

CatBoost ranks furthest from TabPFN and lost because it is much weaker (alone
0.94800 against 0.96026 for TabPFN). The v3 checkpoints rank about as differently
as Kumo and lost. Kumo alone was never submitted, so its own board score is not
known. The blend gain is the evidence that it is strong enough to help and
different enough to matter. The estimator count is not the reason: Kumo runs 8
and TabPFN with 8 lost (row 65).

### 27.4 What transfers, updated

| Category | Record on the board |
|---|---|
| Settings, seeds, averaging (xgboost) | Transferred at about half the offline size, until the settings direction ran out (§11) |
| Same model on recent windows, blended | Gained, in xgboost and again in TabPFN |
| A pretrained tabular model in place of xgboost | **Gained**: TabPFN 0.96026 alone, 0.96825 as a tuned blend |
| More of the same TabPFN (bags, weights, context size, estimators, checkpoints, windows) | Plateau at 0.968; 8 estimators 0.96694, v3 checkpoints 0.96701 |
| Feature-set changes | No gain: graph features, dropped shifted features, within-day ranks, and now six time markers dropped (0.96402) all lost; labelled-neighbour features tied; exact-match counts no gain offline |
| Post-processing | No gain: within-day rank both ways, block offset (optimum at zero) |
| Partner next to TabPFN | 1 for 5: Kumo gained; xgboost, CatBoost, TabICLv2 and TabPFN v3 lost |

§10.4 said to stop testing model families. That held for the seven tried then
(trees, linear, a neural net). It did not hold for pretrained tabular models:
TabPFN beat xgboost, and a second pretrained model from a different family moved
the TabPFN plateau. The plateau was a ceiling for one family, not for the task.

---

## 28. Where this could still go

Written on 2026-10-23; §29 has what happened. The deadline is 2026-10-24 15:55
UTC. Three submissions a day, two final selections.

- **Kumo weight.** Only .25 was tried. Going from 0 to .25 gained 0.0034, and
  nothing says the optimum is at .25. A .35 or .40 share costs no GPU time: both
  parts already exist and `src/blend.py` combines them in seconds. Anything
  under ~0.001 on the board cannot be told from noise (§13), so only a clear
  gain would justify switching a final selection.
- **More Kumo seeds.** Seeds agree at 0.98, so more runs of the same recipe are
  expected to add little. They are cheap, about 2 minutes each at 8k.
- **A Kumo 12k days-18+ context.** The TabPFN blend's recent share uses 8k and
  12k bags. The Kumo component has 8k and the full 14,380 rows, and no 12k.

All three are tuning around one result. Anything touching the feature set has
no record of gaining.

---

## 29. 2026-10-24 — the Kumo weight, last day

The Kumo component was rebuilt from 22 runs (10 seeds each at days-18+ 8k and
all-days 12k, plus the two full-context runs; same .75 / .25 split). It ranks the
test rows like the 12-run component (Spearman 0.9995), so the weight is the only
real change. The day's three submissions searched the weight:

| # | Kumo share | Public score | vs 0.96825 |
|---|---|---|---|
| 67 | .25 (12 runs, 2026-10-23) | 0.97162 | +0.0034 |
| 70 | .33 | 0.97229 | +0.0040 |
| 68 | .40 | 0.97280 | +0.0046 |
| 69 | **.55** | **0.97357** | **+0.0053** |

Every step up gained, so the optimum is at .55 or beyond; no submissions were
left to look further. Kumo now carries more weight than TabPFN. The four points
lie on one smooth rise, which makes it unlikely to be public-sample noise: each
step is a small change to the same two parts. Standing at the end: 3rd, behind
0.97582 and 0.97446.
