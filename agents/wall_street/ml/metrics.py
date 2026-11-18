"""Ranking metrics: how quant teams judge a stock-selection signal.

Accuracy asks "was each call right?". A stock picker cares about something else: "do the
stocks I rank highest actually do better than the ones I rank lowest?" Two standard ways
to measure that:

* Rank IC (information coefficient): on each date, the Spearman correlation between the
  model's scores and the stocks' realized forward returns, averaged over dates. 0 = no
  skill. Real-world signals live around 0.02-0.05; above ~0.10 on price data, suspect a leak.

* Quantile spread: on each date, sort stocks by score into 5 buckets (quintiles) and
  compare the average forward return of the top bucket with the bottom one. Also check
  that the buckets are roughly MONOTONIC (Q1 < Q2 < ... < Q5) - a real signal ranks the
  whole universe, not just one lucky extreme.

The statistics trap (important!): with a 21-day horizon, the forward returns measured on
consecutive days overlap by 20 days, so daily IC values are NOT independent. Computing a
t-statistic over all daily values would overstate significance by roughly sqrt(horizon).
So the t-stat uses only every `horizon`-th date: non-overlapping periods, honest count.
"""

import numpy as np
import pandas as pd


def _non_overlapping(dates: pd.Index, horizon: int) -> pd.Index:
    return pd.Index(np.sort(dates.unique()))[::max(horizon, 1)]


def daily_rank_ic(oos: pd.DataFrame) -> pd.Series:
    """oos: rows with date, prob, fwd_return. Returns one Spearman IC per date."""
    df = oos.dropna(subset=["prob", "fwd_return"])
    g = df.groupby("date")
    # Spearman = Pearson correlation of the ranks. Done by hand (vectorized) rather than
    # via groupby().apply(corr), which is ~100x slower on 90k rows.
    r_p = g["prob"].rank()
    r_y = g["fwd_return"].rank()
    d_p = r_p - r_p.groupby(df["date"]).transform("mean")
    d_y = r_y - r_y.groupby(df["date"]).transform("mean")
    num = (d_p * d_y).groupby(df["date"]).sum()
    den = np.sqrt((d_p ** 2).groupby(df["date"]).sum() * (d_y ** 2).groupby(df["date"]).sum())
    return (num / den.replace(0, np.nan)).dropna()


def rank_ic_summary(oos: pd.DataFrame, horizon: int) -> dict:
    ic = daily_rank_ic(oos)
    if ic.empty:
        return {"rank_ic": 0.0, "ic_tstat": 0.0, "ic_positive_pct": 0.0, "ic_periods": 0}
    sample = ic.loc[ic.index.isin(_non_overlapping(ic.index, horizon))]
    tstat = sample.mean() / sample.std() * np.sqrt(len(sample)) if len(sample) > 1 and sample.std() > 0 else 0.0
    return {
        "rank_ic": round(float(ic.mean()), 4),
        "ic_tstat": round(float(tstat), 2),            # |t| > 2 ~ statistically significant
        "ic_positive_pct": round(float((ic > 0).mean()) * 100, 1),
        "ic_periods": len(sample),                     # independent periods behind the t-stat
    }


def quantile_spread(oos: pd.DataFrame, horizon: int, buckets: int = 5) -> dict:
    """Average forward return per score bucket, on non-overlapping dates."""
    df = oos.dropna(subset=["prob", "fwd_return"])
    df = df[df["date"].isin(_non_overlapping(pd.Index(df["date"]), horizon))]
    g = df.groupby("date")["prob"]
    # Need 1+ stock per bucket, and real differences in score: if every score is equal (the
    # naive baseline), buckets would just follow tie-break order - a meaningless spread.
    df = df[(g.transform("size") >= buckets) & (g.transform("nunique") > 1)].copy()
    if df.empty:
        return {}
    # rank(method="first") breaks ties, so equal scores still spread evenly over buckets.
    pct = df.groupby("date")["prob"].rank(method="first", pct=True)
    df["bucket"] = np.ceil(pct * buckets).clip(1, buckets).astype(int)
    by_bucket = df.groupby(["date", "bucket"])["fwd_return"].mean().unstack()
    spread = by_bucket[buckets] - by_bucket[1]
    means = by_bucket.mean()
    return {
        "bucket_mean_fwd_return_pct": {f"Q{b}": round(float(means[b]) * 100, 3) for b in means.index},
        "top_minus_bottom_pct": round(float(spread.mean()) * 100, 3),   # per horizon period
        "top_beats_bottom_pct_of_periods": round(float((spread > 0).mean()) * 100, 1),
        "monotonic": bool(means.is_monotonic_increasing),
    }
