# Progress report

BT4012 Kaggle competition `bt-4012-competition-2026` — detecting illicit Bitcoin
transactions in the Elliptic dataset, scored by ROC AUC.

**Standing: 1st place, public score 0.95333.** Seven submissions.
Last updated 2026-09-14.

---

## 1. Result

| | Public score |
|---|---|
| **This model** | **0.95333** |
| 2nd place | 0.94938 |
| 3rd place | 0.94890 |
| Organisers' baseline | 0.92119 |

The final model is gradient-boosted decision trees (XGBoost) on the 165 supplied
features, labelled rows only, five random seeds rank-averaged.
`src/model_tuned.py`.

| Setting | Value |
|---|---|
| `max_depth` | 10 |
| `colsample_bytree` | 0.5 |
| `learning_rate` | 0.1 |
| `scale_pos_weight` | 7.57 |
| `min_child_weight` | 8 |
| trees | 68 |
| seeds | 5, rank-averaged |

No feature engineering, no external data, no graph features, no use of the
110,537 unlabelled rows. Every one of those was tried and none helped.

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
| 7 | + `scale_pos_weight` 7.57, `min_child_weight` 8 | 0.9101 | **0.95333** | **+0.0052** |

Three of the seven improved the score. All three were changes to model
*settings*. Every change to the feature set or the model family failed.

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
36–49, that count is **exactly 0 for every training row and non-zero for every
test row**:

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

## 9. Where this could still go

Remaining directions, all in the only category that has transferred:

- **Widen the seed average** from 5 to 20. Pure variance reduction, and the one
  change that has never backfired.
- **Sweep `subsample`, `reg_lambda` and `gamma` in combination** with the
  current settings. Never explored together, and every gain in this project has
  come from combinations rather than single settings.

Realistic remaining headroom is roughly +0.005 to +0.010. The top of the
leaderboard spans 0.0022, so that is enough to matter, but it is tuning rather
than a breakthrough.

Anything touching the feature set or the model family has a 0-for-3 record and
no offline measurement can screen it first. Such a change is a submission spent
to buy information, and should be described that way.
