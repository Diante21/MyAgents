# How the ML Training Works (step by step)

What happens when you run `python -m agents.wall_street.ml.train`, following `main()` in
[train.py](../agents/wall_street/ml/train.py). The numbers come from the first real
training run (2026-11-15, model version `20261115-223303`).

## The question the model answers

> "Given everything about a stock's price behavior up to today's close, will it **beat the
> S&P 500 (SPY)** over the next **5 trading days**?"

This is **binary classification**: the answer is 1 (beats SPY) or 0 (doesn't), and the
model outputs a probability between 0 and 1.

---

## Step 1: Download raw data ([dataset.py](../agents/wall_street/ml/dataset.py), `load_prices`)

For each of the 40 stocks in `DEFAULT_UNIVERSE`, plus SPY, it downloads about 10 years of
daily prices (Open, High, Low, Close, Volume) from Yahoo Finance.

- Each file is cached in `data/raw/<TICKER>.csv`, so retraining doesn't download again.
  Use `--refresh` to force a fresh download.
- SPY is needed for two things: market-context features, and the label ("beat SPY").

## Step 2: Build features ([features.py](../agents/wall_street/ml/features.py), `FeatureBuilder.build`)

The model can't learn from raw prices: $150 means something different for every stock. So
each day is turned into 20 **comparable** numbers:

| Group | Features | What they capture |
|---|---|---|
| Momentum | returns over 1, 5, 21, 63 and 126 days | Has it been going up or down, and over what timescale? |
| Volatility | 21- and 63-day volatility, and their ratio | How jumpy is it, and is it getting jumpier? |
| Oscillators | RSI, MACD histogram as a % of price | Overbought or oversold, momentum shifts |
| Trend | % distance from the 20, 50 and 200-day averages; drawdown from the 1-year high | Where the price sits relative to its trend |
| Volume | volume z-score (how unusual today's volume is) | Unusual activity, often news |
| Market | SPY's 5- and 21-day returns, SPY volatility | What the whole market is doing |
| Relative | stock return minus SPY return over 21 and 63 days | Is it leading or lagging the market? |

**The rule everything depends on:** the feature for day *t* uses only data up to day *t*'s
close. Every feature comes from a rolling window that *ends* at *t*. Breaking this rule is
called **look-ahead bias**, and it's the main reason stock models look great in testing and
fail live. `test_features_do_not_use_the_future` checks it by rewriting future prices and
confirming that past features don't change.

The first ~260 days of each stock are dropped, because the 200-day average and 252-day
drawdown need that much history.

## Step 3: Create labels ([dataset.py](../agents/wall_street/ml/dataset.py), `make_labels`)

For each day *t*:

```
stock_return = close[t+5] / close[t] - 1
spy_return   = SPY[t+5] / SPY[t] - 1
label = 1 if stock_return - spy_return > 0 else 0
```

- The label **looks forward** 5 days. That's the only place future data is allowed,
  because it's the answer the model is learning to predict.
- The last 5 days get no label, since that future hasn't happened yet, so they're dropped.
- Two more forward-looking columns are kept for the backtest (`fwd_return`,
  `next_day_return`). They are **never** features.

**Why "beat SPY" and not "go up"?** Over 2017–2026 stocks went up on most 5-day stretches,
so a model that always says "up" would look skilled. "Beat SPY" removes the market's
overall direction and gives a near 50/50 split. The data had a **50.3% base rate**.

## Step 4: Stack everything into one table

All 40 stocks are combined into one **panel**: one row per (date, stock).

```
date        ticker  ret_1d  ret_5d  ...  rel_strength_63d  label
2017-11-13  AAPL    0.004   0.012   ...  0.031             1
2017-11-13  AMD    -0.011   0.025   ...  -0.044            0
...
```

The run produced **90,320 rows**. One model learns from all stocks together. That gives it
far more examples than a separate model per stock, and it learns general patterns ("stocks
that fell hard tend to...") rather than quirks of one ticker.

## Step 5: Walk-forward validation ([trainer.py](../agents/wall_street/ml/trainer.py))

This is the most important step for honest results. The dates are split like this:

```
2017 ─────────────── 2021 │ gap │ test 1 (2021-22)
2017 ──────────────────── 2022 │ gap │ test 2 (2022-23)
2017 ───────────────────────── 2023 │ gap │ test 3 (2023-24)
2017 ────────────────────────────── 2024 │ gap │ test 4 (2024-25)
2017 ─────────────────────────────────── 2025 │ gap │ test 5 (2025-26)
```

For each of the 5 folds:
1. **Train** on every day before the test window. This is where `model.fit(X, y)` runs and
   the model actually learns.
2. **Skip a 5-day gap** (the "embargo"). A training row from 3 days before the test window
   has a label that looks 5 days ahead, *into* the test window. Without the gap, test-period
   information would leak into training.
3. **Predict** on the test window, using data the model has never seen.
4. **Score** it: accuracy, AUC and log loss, plus what a naive guess would have scored.

**Why not a random 80/20 split?** Random splits put 2025 days in training and 2019 days in
testing, so the model would learn from the future to predict the past. Results would look
great and mean nothing.

The test windows don't overlap, so together they give one **out-of-sample** prediction for
every day from mid-2021 to Nov 2026, each made by a model that only saw earlier data.
That's exactly the situation you'd be in live.

## Step 6: Compare three models ([models.py](../agents/wall_street/ml/models.py))

All three go through the same walk-forward process:

| Model | How it works | Why it's included |
|---|---|---|
| **Naive baseline** | Ignores features and always predicts the training base rate | The bar to beat. Any "skill" below this is fake. |
| **Logistic regression** | A weighted sum of the features, passed through a sigmoid | Simple, hard to overfit, easy to interpret |
| **Gradient boosting** | Hundreds of small decision trees, each correcting the previous ones | Can learn interactions, like "momentum works unless volatility is spiking" |

Both real models are heavily **regularized** (small `C` for logistic regression; shallow
trees and big leaves for boosting). Stock data is mostly noise, and a flexible model will
happily memorize it.

**How to read the metrics:**
- **Accuracy:** % of correct calls. 50% is a coin flip on balanced data.
- **AUC:** how well the model *ranks* stocks (does a higher probability really mean better
  odds?). 0.5 is random, 1.0 is perfect. This is the main metric because it doesn't depend
  on a 0.5 cutoff.
- **Log loss:** penalizes confident wrong predictions. Lower is better, and 0.693 is pure
  guessing.

Results:

| Model | Accuracy | AUC |
|---|---|---|
| naive baseline | 50.0% | 0.500 |
| logistic | 50.2% | 0.508 |
| gradient boosting | 50.2% | 0.507 |

Logistic won narrowly. Per fold, its AUC went 0.522, 0.512, **0.498**, 0.501, 0.506, so one
fold was worse than random and the edge faded in recent years. That's a sign the pattern is
weak or disappearing.

## Step 7: Backtest ([backtest.py](../agents/wall_street/ml/backtest.py))

The metrics are statistics; the backtest asks whether the predictions would have made
money. Using the out-of-sample predictions from Step 5:

- Each day, for each stock: **hold it if probability > 0.5, otherwise hold cash.**
- The signal from day *t*'s close earns the return from *t* to *t+1*, so there's no peeking.
- Each switch in or out costs 5 basis points (0.05%) for commissions, spread and slippage.
- The 40 stock positions are averaged with equal weight and compared to **buy-and-hold of
  the same 40 stocks**.

Result (2021-06-17 → 2026-11-06):

| | Strategy | Buy-and-hold |
|---|---|---|
| Total return | 78% | **145%** |
| Sharpe | 0.90 | **1.12** |
| Max drawdown | **-17%** | -24% |

The strategy lost. Its smaller drawdown only came from sitting in cash 35% of the time, not
from any skill.

## Step 8: Train the final model and save it ([registry.py](../agents/wall_street/ml/registry.py))

1. The winning model type (logistic) is **retrained on all 90,320 rows**, since more data
   makes a better final model.
2. It's saved to `models/price_move/<version>/model.joblib`.
3. `meta.json` stores the feature list (in order), horizon, target, stock list, date range,
   the **walk-forward scores from Step 5**, the backtest, and `beats_baseline` (true only if
   AUC ≥ 0.51).

**Important detail:** the saved scores come from walk-forward testing, *not* from the final
model, because the final model has already seen all the data, so testing it on that data
would be meaningless. The out-of-sample numbers are the honest estimate of how it will do
live.

## Step 9: Using it live ([tools/predict.py](../agents/wall_street/tools/predict.py))

When Claude calls `predict_price_move("NVDA")`:
1. It loads the latest model and `meta.json`.
2. It downloads 2 years of NVDA and SPY prices.
3. It runs **the same `FeatureBuilder`** from Step 2. Using different feature code in
   training and live use is a classic bug called **train/serve skew**.
4. It takes the newest row, the one for today, and gets a probability.
5. It returns the probability **together with the track record from `meta.json`**, so
   Claude sees "0.595, but this model hasn't beaten a coin flip".

---

## The full flow

```
prices (Yahoo) ──► features (20 per day, past-only)
                        │
               labels (beat SPY in 5 days?, future-looking)
                        │
                 panel: 90,320 rows
                        │
        walk-forward × 3 models ──► out-of-sample scores ──► pick best
                        │                     │
                        │                 backtest
                        ▼                     │
        retrain best on all data ──► save model + honest scores
                                              │
                         predict_price_move tool ──► Claude
```

## Interview-ready takeaways

1. **Leakage is the number-one risk** in financial ML. Prevent it with past-only features,
   forward-only labels, walk-forward splits and an embargo gap, then **test for it**.
2. **Always compare against a naive baseline.** "52% accuracy" means nothing without
   knowing the base rate.
3. **Pick the target carefully.** "Beat SPY" versus "go up" changed what counts as skill.
4. **Report out-of-sample results**, never results on the training data.
5. **A backtest with costs** is the real test. Good statistics don't guarantee profit.
6. **A negative result is still a result.** This pipeline told the truth instead of a
   flattering number, which is what a good one should do.

---

# Experiment 1: cross-sectional ranks, longer horizon, ranking metrics (2026-11-16)

**Goal:** get past the ~50% accuracy without a bigger model, by changing *the question*
and *how the inputs are expressed*. Change one thing per run so each effect can be measured.

## What was added

| Piece | Where | What it does |
|---|---|---|
| `--features rank` | [features.py](../agents/wall_street/ml/features.py) `add_cross_sectional_ranks`, `build_panel` | Each stock feature becomes its percentile among all stocks **on the same date**, centered at 0 (-0.5 lowest, +0.5 highest). Market features stay raw, since they're the same for every stock that day. |
| `--target xs` | [dataset.py](../agents/wall_street/ml/dataset.py) | Label = 1 if the stock beats the **median stock** of the universe over the horizon. Exactly 50/50 every day. |
| Rank IC + quintile spread | [metrics.py](../agents/wall_street/ml/metrics.py) | "Does a higher score really mean a better stock?" The t-stat uses only **non-overlapping** periods. |
| Top-20% backtest | [backtest.py](../agents/wall_street/ml/backtest.py) `run_top_bucket` | Hold the top-scored 20%, rebalance every `horizon` days, always fully invested, so it measures *selection* skill. |
| Stricter edge check | [train.py](../agents/wall_street/ml/train.py) | `beats_baseline` now needs AUC ≥ 0.51 **and** IC t-stat ≥ 2. |
| `--no-save` | train.py | Run experiments without replacing the agent's model. |
| Live ranking | [tools/predict.py](../agents/wall_street/tools/predict.py) | A rank model ranks the stock against its ~40-stock universe live, using the same `build_panel()` as training (so there's no train/serve skew). |

**Why ranks?** Raw values drift with the market regime. In 2020 every stock was "high
volatility", so a rule like "vol > 40% is bad" learned in 2019 breaks. A percentile is
always spread uniformly from -0.5 to +0.5, which makes it stable over time and robust to
outliers. Ranks only compare stocks on the same day, so they can't leak the future (this
is tested in `test_ranks_do_not_use_the_future`).

**Why the t-stat uses non-overlapping periods:** with a 21-day horizon, today's and
tomorrow's forward returns share 20 days, so daily IC values are not independent. Counting
all ~1,350 days would overstate significance by about √21 ≈ 4.6×. Only every 21st date is
used, which leaves **65 honest periods**.

## Results (walk-forward, out-of-sample 2021-06 → 2026-10, best model = logistic)

| Run | Features | Target | Horizon | AUC | Rank IC | IC t-stat | Folds with AUC > 0.5 | Top − bottom quintile per period | Top 20% vs. hold-all |
|---|---|---|---|---|---|---|---|---|---|
| A (old setup) | raw | excess | 5 | 0.5075 | 0.010 | 1.15 | 4 / 5 | +0.02% | 138% vs. 145% |
| B | raw | excess | **21** | 0.4983 | 0.025 | 0.29 | 2 / 5 | −0.49% | 102% vs. 145% |
| C | **rank** | excess | 21 | 0.5080 | 0.027 | 0.68 | 4 / 5 | +0.45% | 193% vs. 145% |
| D | rank | **xs** | 21 | **0.5148** | **0.029** | 0.71 | **5 / 5** | +0.53% | 189% vs. 145% |

To reproduce a row: `python -m agents.wall_street.ml.train --no-save --features <f> --target <t> --horizon <h>`.

## What we learned

1. **The longer horizon alone did NOT help (A → B).** With raw features, the 21-day model
   got *worse* and unstable (fold AUCs from 0.52 down to 0.47). A longer horizon gives
   fundamentals time to matter, but our features are still price-only.
2. **Ranks are what helped (B → C).** Same data and target, but every fold except one now
   has a positive IC, and the top quintile beats the bottom one instead of losing to it.
   That's the regime-drift argument showing up in practice.
3. **The cross-sectional target tidied it up (C → D).** Best AUC (0.515), every fold above
   0.50, best IC. Asking a cleaner question gave a cleaner answer.
4. **But it is NOT statistically proven.** IC 0.029 is in the "typical real signal" range,
   yet the t-stat is 0.71, far below 2. With only 40 stocks, each period's IC is very noisy
   (std roughly 0.3), and 65 independent periods can't tell 0.03 apart from 0. **This is the
   strongest argument for Experiment 2 (more stocks)**: IC noise shrinks roughly like
   1/√(number of stocks).
5. **Read the backtest carefully.** The top-20% portfolio returned 189% vs. 145%, but with
   higher volatility (22.8% vs. 15.9%), a deeper drawdown (−31% vs. −24%) and a *lower*
   Sharpe (0.98 vs. 1.14). Its most frequent picks were AMD, NVDA and AVGO, and its picks
   lean toward stronger 6-month momentum and slightly higher volatility. Much of the
   extra return is probably **riding the 2023–25 semiconductor run with an 8-stock
   portfolio**, not proven skill. Return without risk adjustment is not evidence.
6. **The quintiles aren't monotonic** (Q1 1.22%, Q2 1.72%, Q3 1.06%, Q4 1.65%, Q5 1.75% per
   21 days). A strong signal ranks the whole universe in order; this one mostly separates
   the extremes.

## What the rank model learned (logistic coefficients, run D)

Largest weights: **+0.33 `pct_vs_sma_200`** (above its long-term trend relative to
peers), **+0.15 `rsi_14`**, **−0.15 `ret_126d`**, **−0.11 `drawdown_252d`**. Market
features are ≈ 0, which is correct: they're identical for every stock on a date, so they
can't help choose *between* stocks.

⚠️ Don't read single coefficients too literally. `pct_vs_sma_200`, `ret_126d` and
`drawdown_252d` are strongly correlated (**multicollinearity**), so the model can offset
one against another. Read the group together, roughly: "in a long-term uptrend relative to
peers, but not the most stretched over 6 months". That's a known blend of trend and
mean reversion.

## Decision

The rank/xs/21-day setup is the best configuration so far and the new direction. It still
fails the edge check, so **the agent's saved model was not replaced** (all runs used
`--no-save`). To switch the agent to it anyway (it will still be labelled "no reliable
edge" to Claude):

```bash
python -m agents.wall_street.ml.train --features rank --target xs --horizon 21
```

**Next: Experiment 2.** Expand to ~500 stocks (S&P 500) to shrink IC noise, and check
whether the IC t-stat clears 2. Watch for survivorship bias.

## Interview takeaways from this experiment
- **Represent features so they're stable over time.** Cross-sectional ranks are the
  standard quant trick.
- **Match the target, metric and backtest to the decision.** Picking stocks means a
  ranking target, rank IC, and a long-the-top portfolio.
- **Overlapping labels inflate significance.** Count only independent periods.
- **Ask what the strategy is really holding.** Higher return with higher volatility and a
  lower Sharpe is often just a tilt toward risk, not skill.
