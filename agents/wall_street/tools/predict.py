"""PredictPriceMoveTool: exposes the trained ML model to Claude as a tool.

This is where the two halves of the system meet:
  * the ML model gives a calibrated-ish probability from price patterns,
  * Claude decides how much that probability matters next to fundamentals and news.

The key design choice: the tool returns the model's OUT-OF-SAMPLE track record along with
every prediction. A probability alone ("62% up!") invites over-trust; a probability
next to "this model's AUC is 0.52 and it barely beats a coin flip" lets the agent - and
you - weigh it correctly.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import pandas as pd

from core.tools import Tool, ToolError
from agents.wall_street.data.providers import MarketDataProvider
from agents.wall_street.ml.dataset import MARKET_TICKER
from agents.wall_street.data.providers import DataUnavailable
from agents.wall_street.ml.features import MIN_CROSS_SECTION, MIN_HISTORY_DAYS, FeatureBuilder, build_panel
from agents.wall_street.ml.registry import ModelRegistry
from agents.wall_street.tools import clean_ticker, provider_call
from agents.wall_street.tools.market_data import TICKER_PROP

TARGET_MEANING = {
    "excess": "probability the stock OUTPERFORMS the S&P 500 (SPY) over the horizon",
    "direction": "probability the stock's price is HIGHER at the end of the horizon",
    "xs": "probability the stock OUTPERFORMS THE MEDIAN STOCK of the model's universe "
          "(~40 large US companies) over the horizon",
}


def read_signal(prob: float, beats_baseline: bool) -> str:
    edge = abs(prob - 0.5)
    strength = "no meaningful lean" if edge < 0.02 else "weak lean" if edge < 0.05 else "moderate lean"
    reading = strength if edge < 0.02 else f"{strength} {'positive' if prob > 0.5 else 'negative'}"
    if not beats_baseline:
        # A confident-looking number from a model with no proven edge is still noise.
        reading += " - but this model has NOT beaten a coin flip out of sample, so treat it as noise"
    return reading


class PredictPriceMoveTool(Tool):
    name = "predict_price_move"
    description = (
        "Run the user's own trained machine-learning model on a stock's recent price and "
        "volume patterns to get a probability for its move over the next few trading days, "
        "together with the model's honest out-of-sample track record. It uses only "
        "technical/price features - no fundamentals or news - so treat it as one weak, "
        "short-term signal among many, weighted by its track record."
    )
    input_schema = {
        "type": "object",
        "properties": {"ticker": TICKER_PROP},
        "required": ["ticker"],
        "additionalProperties": False,
    }

    def __init__(self, provider: MarketDataProvider, registry: ModelRegistry):
        self.provider = provider
        self.registry = registry
        self.features = FeatureBuilder()
        self._loaded = None   # load lazily, once

    def _model(self):
        if self._loaded is None:
            try:
                self._loaded = self.registry.load_latest()
            except FileNotFoundError as e:
                raise ToolError(str(e)) from e
        return self._loaded

    def run(self, ticker: str) -> dict[str, Any]:
        t = clean_ticker(ticker)
        model, meta = self._model()

        prices = provider_call(self.provider.get_price_history, t, "2y")
        market = provider_call(self.provider.get_price_history, MARKET_TICKER, "2y")
        if len(prices) < MIN_HISTORY_DAYS:
            raise ToolError(f"{t} has only {len(prices)} days of history; the model needs {MIN_HISTORY_DAYS}.")

        if meta.get("feature_set", "raw") == "rank":
            row = self._ranked_row(t, prices, market, meta["universe"])
        else:
            feats = self.features.build(prices, market)
            if feats.empty:
                raise ToolError(f"Could not compute features for {t}.")
            row = feats.iloc[[-1]].assign(date=feats.index[-1])
        latest = row[meta["features"]]   # same columns, same order as training
        prob = float(model.predict_proba(latest)[0])
        as_of = pd.Timestamp(row["date"].iloc[0])
        oos = meta["oos_metrics"]

        return {
            "ticker": t,
            "as_of": as_of,
            "data_is_stale": (datetime.now() - as_of.to_pydatetime()).days > 5,
            "horizon_trading_days": meta["horizon_days"],
            "what_probability_means": TARGET_MEANING[meta["target"]],
            "probability": prob,
            "reading": read_signal(prob, meta["beats_baseline"]),
            "in_training_universe": t in meta["universe"],
            "model": {
                "type": meta["model"],
                "version": meta["version"],
                "trained_on": meta["trained_on"],
                "out_of_sample_accuracy": oos["accuracy"],
                "naive_baseline_accuracy": oos["baseline_accuracy"],
                "out_of_sample_auc": oos["auc"],
                # Ranking skill (older models were saved before these were measured).
                "out_of_sample_rank_ic": oos.get("rank_ic"),
                "rank_ic_t_stat": oos.get("ic_tstat"),
                "beats_baseline": meta["beats_baseline"],
                "backtest_vs_buy_and_hold": {
                    "strategy_sharpe": meta["backtest"]["strategy"]["sharpe"],
                    "buy_and_hold_sharpe": meta["backtest"]["buy_and_hold"]["sharpe"],
                },
            },
            "how_to_weigh": (
                "AUC 0.50 = coin flip. In daily stock prediction, 0.51-0.53 is typical for a "
                "real but small edge. Rank IC 0 = no skill, 0.02-0.05 = typical real signal; a "
                "t-stat below 2 means the skill can't be told apart from luck. Never let this "
                "override fundamentals or a clear risk."
            ),
            "key_inputs": {k: round(float(row[k].iloc[0]), 4) for k in
                           ("ret_21d", "rel_strength_63d", "rsi_14", "pct_vs_sma_50", "vol_21d")},
        }

    def _ranked_row(self, ticker: str, prices, market, universe: list[str]) -> pd.DataFrame:
        """Rank-feature models compare the stock with the whole training universe on the
        same day, so we need everyone's latest prices. Same build_panel() as training -
        no train/serve skew. First call downloads ~40 histories (cached for 5 minutes)."""
        others = [u for u in universe if u != ticker]

        def fetch(u):
            try:
                return u, self.provider.get_price_history(u, "2y")
            except DataUnavailable:
                return u, None          # one missing stock shouldn't block the prediction

        with ThreadPoolExecutor(max_workers=8) as pool:
            fetched = dict(pool.map(fetch, others))
        prices_by_ticker = {u: df for u, df in fetched.items()
                            if df is not None and len(df) >= MIN_HISTORY_DAYS}
        prices_by_ticker[ticker] = prices

        panel = build_panel(self.features, prices_by_ticker, market)
        mine = panel[panel["ticker"] == ticker].dropna(subset=["ret_1d_xrank"])
        if mine.empty:
            raise ToolError(f"Could not rank {ticker} against the model's universe "
                            f"(need {MIN_CROSS_SECTION}+ stocks with data on the same day).")
        return mine.iloc[[-1]]
