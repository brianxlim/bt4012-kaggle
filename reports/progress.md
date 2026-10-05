# Progress report

BT4012 Kaggle competition `bt-4012-competition-2026` — detecting illicit Bitcoin
transactions in the Elliptic dataset, scored by ROC AUC.

**Standing: 1st place, public score 0.95521.** Eight submissions.
Last updated 2026-09-17.

---

## 1. Result

| | Public score |
|---|---|
| **This model** | **0.95521** |
| 2nd place | 0.95357 |
| 3rd place | 0.95085 |
| Organisers' baseline | 0.92119 |

The final model is gradient-boosted decision trees (XGBoost) on the 165 supplied
features, labelled rows only, five random seeds rank-averaged.
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

Four of the eight improved the score. All four were changes to model *settings*.
Every change to the feature set or the model family failed.

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
