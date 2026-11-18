"""DatasetBuilder: downloads price history for a universe of stocks and builds a
(features, label) table for training.

Label choices (the TARGET is the most important modelling decision):
  "excess"    -> 1 if the stock beats SPY over the next `horizon` days (default).
                 Roughly 50/50 classes, removes the overall market direction, and it's
                 what matters for picking stocks: "is this a better buy than the index?"
  "direction" -> 1 if the stock simply goes up. Easier to understand, but in a rising
                 market ~55% of labels are 1, so "always predict up" looks skilful.
  "xs"        -> 1 if the stock beats the MEDIAN stock of the universe over the horizon
                 ("cross-sectional"). Exactly 50/50 every single day, and it asks the
                 cleanest stock-picking question: "which of these will do better than the
                 others?" Pairs naturally with the "rank" feature set.

The table is a "panel": one row per (date, ticker), so one model learns from all stocks.
"""

from pathlib import Path
import logging

import pandas as pd

from agents.wall_street.data.providers import DataUnavailable, MarketDataProvider
from agents.wall_street.ml.features import FEATURE_COLUMNS, FeatureBuilder, build_panel

log = logging.getLogger(__name__)

MARKET_TICKER = "SPY"
TARGETS = ("excess", "direction", "xs")

# ~40 large US companies across sectors. WARNING - survivorship bias: these are
# companies that are big TODAY. A 2016 investor couldn't have known that, so backtests
# on this list look better than reality. A proper study uses historical index members.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ORCL", "CRM", "ADBE",   # tech
    "AMD", "INTC", "QCOM", "CSCO", "IBM",
    "JPM", "BAC", "GS", "V", "MA",                                                   # financials
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV",                                       # healthcare
    "XOM", "CVX",                                                                    # energy
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "HD",                            # consumer
    "CAT", "BA", "GE", "UNP",                                                        # industrials
]


class DatasetBuilder:
    def __init__(self, provider: MarketDataProvider, raw_dir: Path, period: str = "10y"):
        self.provider = provider
        self.raw_dir = raw_dir
        self.period = period
        self.features = FeatureBuilder()

    # ---- raw data (cached on disk so re-training doesn't re-download) -------------
    def load_prices(self, ticker: str, refresh: bool = False) -> pd.DataFrame:
        path = self.raw_dir / f"{ticker.replace('^', '_')}.csv"
        if path.exists() and not refresh:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        df = self.provider.get_price_history(ticker, self.period)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        return df

    # ---- labels -----------------------------------------------------------------
    @staticmethod
    def make_labels(close: pd.Series, market_close: pd.Series, horizon: int, target: str) -> pd.Series:
        """Label at day t looks FORWARD from t to t+horizon. The last `horizon` days have
        no label yet (the future hasn't happened) and come out as NaN."""
        fwd = close.shift(-horizon) / close - 1
        if target == "excess":
            mkt = market_close.reindex(close.index).ffill()
            fwd = fwd - (mkt.shift(-horizon) / mkt - 1)
        elif target != "direction":
            raise ValueError(f"Unknown target {target!r}")
        label = (fwd > 0).astype(float)
        label[fwd.isna()] = float("nan")
        return label

    # ---- the full table -----------------------------------------------------------
    def build(self, tickers: list[str], horizon: int = 5, target: str = "excess",
              refresh: bool = False) -> pd.DataFrame:
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}")
        market = self.load_prices(MARKET_TICKER, refresh)
        prices_by_ticker, outcomes = {}, []
        for ticker in tickers:
            try:
                prices = self.load_prices(ticker, refresh)
            except DataUnavailable as e:
                log.warning("Skipping %s: %s", ticker, e)
                continue
            prices_by_ticker[ticker] = prices
            close = prices["Close"]
            outcomes.append(pd.DataFrame({
                "date": close.index,
                "ticker": ticker,
                # "xs" labels need every stock's return first, so start from "excess".
                "label": self.make_labels(close, market["Close"], horizon,
                                          "excess" if target == "xs" else target).values,
                # Forward returns kept for metrics + backtest (NOT features - the future!).
                "fwd_return": (close.shift(-horizon) / close - 1).values,
                "next_day_return": (close.shift(-1) / close - 1).values,
            }))
        if not prices_by_ticker:
            raise RuntimeError("No data downloaded - check your internet connection.")

        # Features (raw + cross-sectional ranks) are built for ALL dates, including the
        # last `horizon` days without labels, so ranks match what the live tool computes.
        panel = build_panel(self.features, prices_by_ticker, market)
        data = panel.merge(pd.concat(outcomes), on=["date", "ticker"], how="left")

        if target == "xs":
            median = data.groupby("date")["fwd_return"].transform("median")
            data["label"] = (data["fwd_return"] > median).astype(float)
            data.loc[data["fwd_return"].isna(), "label"] = float("nan")

        data = data.dropna(subset=["label"])
        for ticker, n in data.groupby("ticker").size().items():
            log.info("  %-6s %5d rows", ticker, n)
        return data.sort_values(["date", "ticker"]).reset_index(drop=True)


def split_xy(data: pd.DataFrame, columns: list[str] = FEATURE_COLUMNS) -> tuple[pd.DataFrame, pd.Series]:
    return data[columns], data["label"].astype(int)
