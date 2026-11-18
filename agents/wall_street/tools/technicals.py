"""Technical indicators.

The math lives in `compute_indicators()`, a pure function (DataFrame in, dict out) with
no network access, so it's easy to unit-test and you'll reuse it as ML features in
phase 3. The Tool class is just a thin wrapper around it.
"""

from typing import Any

import pandas as pd

from core.tools import Tool, ToolError
from agents.wall_street.data.providers import MarketDataProvider
from agents.wall_street.tools import clean_ticker, provider_call
from agents.wall_street.tools.market_data import TICKER_PROP


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing). >70 often read as overbought, <30 oversold."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "histogram": line - sig})


def compute_indicators(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 30:
        raise ValueError("Need at least 30 bars of history for indicators")
    close, volume = df["Close"], df["Volume"]
    last = float(close.iloc[-1])

    smas = {}
    for window in (20, 50, 200):
        if len(close) >= window:
            value = float(close.rolling(window).mean().iloc[-1])
            smas[f"sma_{window}"] = value
            smas[f"pct_vs_sma_{window}"] = (last / value - 1) * 100

    m = macd(close).iloc[-1]
    rsi_value = float(rsi(close).iloc[-1])
    trend = None
    if "sma_50" in smas and "sma_200" in smas:
        trend = "golden cross regime (50d > 200d)" if smas["sma_50"] > smas["sma_200"] else \
                "death cross regime (50d < 200d)"

    return {
        "last_close": last,
        **smas,
        "rsi_14": rsi_value,
        "rsi_reading": "overbought" if rsi_value > 70 else "oversold" if rsi_value < 30 else "neutral",
        "macd": float(m["macd"]),
        "macd_signal": float(m["signal"]),
        "macd_histogram": float(m["histogram"]),
        "macd_reading": "bullish (above signal)" if m["histogram"] > 0 else "bearish (below signal)",
        "trend_regime": trend,
        "volume_vs_20d_avg": float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]),
        "pct_from_period_high": (last / float(close.max()) - 1) * 100,
        "pct_from_period_low": (last / float(close.min()) - 1) * 100,
    }


class ComputeTechnicalsTool(Tool):
    name = "compute_technicals"
    description = (
        "Compute technical indicators from the last year of daily prices: 20/50/200-day "
        "moving averages and the price's distance from them, RSI(14), MACD(12,26,9), trend "
        "regime, and volume vs. its 20-day average. Includes a plain-English reading of each. "
        "Technicals describe momentum and trend, not company value."
    )
    input_schema = {
        "type": "object",
        "properties": {"ticker": TICKER_PROP},
        "required": ["ticker"],
        "additionalProperties": False,
    }

    def __init__(self, provider: MarketDataProvider):
        self.provider = provider

    def run(self, ticker: str) -> dict[str, Any]:
        t = clean_ticker(ticker)
        df = provider_call(self.provider.get_price_history, t, "1y")
        try:
            return {"ticker": t, "as_of": df.index[-1], **compute_indicators(df)}
        except ValueError as e:
            raise ToolError(f"{t}: {e}") from e
