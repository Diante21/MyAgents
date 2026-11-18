"""FeatureBuilder: turns raw daily prices into model inputs.

THE rule of financial ML: a feature for day t may only use information available at
the close of day t. Breaking it ("look-ahead bias") produces models that look brilliant
in testing and fail live. Every feature here is built from rolling windows that END at
t, and tests/test_ml.py checks this by changing future prices and asserting that past
features don't move.

The same class is used for training AND for live predictions (tools/predict.py). Using
two different implementations is a classic bug called train/serve skew.

Two feature sets (pick with `train.py --features`):
  "raw"  -> the 20 numbers above, per stock. "NVDA is up 8% this month."
  "rank" -> each stock-specific feature turned into its PERCENTILE among all stocks on the
            same date, centered on 0 (-0.5 = lowest, +0.5 = highest). "NVDA's 1-month
            return is in the top 10% of the universe today."
Why ranks help: raw values drift with the market regime (in 2020 every stock had huge
volatility), so a rule learned in one year may not transfer. Ranks are always uniform
from -0.5 to +0.5, which makes them stable over time and robust to outliers. They pair
naturally with a cross-sectional target ("will this stock beat the median stock?").
Ranks only compare stocks on the SAME date, so they can't leak the future.
"""

import numpy as np
import pandas as pd

from agents.wall_street.tools.technicals import macd, rsi

# Order matters: the model expects columns in exactly this order.
FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d",
    "vol_21d", "vol_63d", "vol_ratio",
    "rsi_14", "macd_hist_pct",
    "pct_vs_sma_20", "pct_vs_sma_50", "pct_vs_sma_200",
    "drawdown_252d", "volume_z_20d",
    "mkt_ret_5d", "mkt_ret_21d", "mkt_vol_21d", "rel_strength_21d", "rel_strength_63d",
]

# Market-wide features are identical for every stock on a date, so ranking them would
# give a constant. They stay raw in the "rank" set (they describe the regime).
MARKET_COLUMNS = ["mkt_ret_5d", "mkt_ret_21d", "mkt_vol_21d"]
STOCK_COLUMNS = [c for c in FEATURE_COLUMNS if c not in MARKET_COLUMNS]
RANK_COLUMNS = [f"{c}_xrank" for c in STOCK_COLUMNS] + MARKET_COLUMNS
FEATURE_SETS = {"raw": FEATURE_COLUMNS, "rank": RANK_COLUMNS}

# How much history one prediction needs (longest window above + a margin).
MIN_HISTORY_DAYS = 260
# Percentiles among 3 stocks are meaningless; ranks need a real cross-section.
MIN_CROSS_SECTION = 10


class FeatureBuilder:
    def build(self, prices: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
        """prices/market: OHLCV DataFrames indexed by date (market = SPY).
        Returns one row per date with FEATURE_COLUMNS; early rows without enough
        history are dropped."""
        close = prices["Close"]
        volume = prices["Volume"].replace(0, np.nan)
        mkt = market["Close"].reindex(close.index).ffill()
        daily = close.pct_change()
        mkt_daily = mkt.pct_change()

        f = pd.DataFrame(index=close.index)
        # Momentum: past returns over several horizons.
        for days in (1, 5, 21, 63, 126):
            f[f"ret_{days}d"] = close.pct_change(days)
        # Risk: recent volatility, and whether it's rising (ratio > 1) or falling.
        f["vol_21d"] = daily.rolling(21).std() * np.sqrt(252)
        f["vol_63d"] = daily.rolling(63).std() * np.sqrt(252)
        f["vol_ratio"] = f["vol_21d"] / f["vol_63d"]
        # Oscillators, scaled so they're comparable across stocks with different prices.
        f["rsi_14"] = rsi(close) / 100
        f["macd_hist_pct"] = macd(close)["histogram"] / close
        # Trend: distance from moving averages.
        for window in (20, 50, 200):
            f[f"pct_vs_sma_{window}"] = close / close.rolling(window).mean() - 1
        f["drawdown_252d"] = close / close.rolling(252).max() - 1
        # Unusual volume often accompanies news.
        log_vol = np.log(volume)
        f["volume_z_20d"] = (log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std()
        # Market context: what the whole market is doing, and the stock relative to it.
        f["mkt_ret_5d"] = mkt.pct_change(5)
        f["mkt_ret_21d"] = mkt.pct_change(21)
        f["mkt_vol_21d"] = mkt_daily.rolling(21).std() * np.sqrt(252)
        f["rel_strength_21d"] = f["ret_21d"] - f["mkt_ret_21d"]
        f["rel_strength_63d"] = f["ret_63d"] - mkt.pct_change(63)

        f = f[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        return f.dropna()


def add_cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """panel: one row per (date, ticker) with a 'date' column and the raw STOCK_COLUMNS.
    Adds '<feature>_xrank' columns: percentile within that date, centered on 0. Dates
    with fewer than MIN_CROSS_SECTION stocks get NaN ranks (and are dropped later)."""
    by_date = panel.groupby("date")
    ranks = by_date[STOCK_COLUMNS].rank(pct=True) - 0.5
    ranks.columns = [f"{c}_xrank" for c in STOCK_COLUMNS]
    ranks[by_date["ticker"].transform("size") < MIN_CROSS_SECTION] = np.nan
    return panel.join(ranks)


def build_panel(builder: FeatureBuilder, prices_by_ticker: dict[str, pd.DataFrame],
                market: pd.DataFrame) -> pd.DataFrame:
    """Raw features for many stocks stacked into one table, plus cross-sectional ranks.
    Used by BOTH the dataset builder (training) and the prediction tool (live), so the
    ranks are computed identically in both places."""
    frames = []
    for ticker, prices in prices_by_ticker.items():
        feats = builder.build(prices, market)
        feats["ticker"] = ticker
        frames.append(feats)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index()
    return add_cross_sectional_ranks(panel)
