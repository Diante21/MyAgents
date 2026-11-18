"""Price and quote tools."""

from typing import Any

from core.tools import Tool
from agents.wall_street.data.providers import MarketDataProvider
from agents.wall_street.tools import clean_ticker, provider_call

TICKER_PROP = {"type": "string", "description": "Ticker symbol, e.g. AAPL, MSFT, BRK-B, ^GSPC"}
PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]

# Broad-market gauges used by the market overview tool.
MARKET_GAUGES = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF",
    "DIA": "Dow Jones ETF",
    "IWM": "Russell 2000 small caps ETF",
    "^VIX": "Volatility index (fear gauge)",
    "^TNX": "10-year Treasury yield (x10)",
    "GLD": "Gold ETF",
    "UUP": "US dollar index ETF",
}


def pct_change(series, periods: int) -> float | None:
    if len(series) <= periods:
        return None
    return float(series.iloc[-1] / series.iloc[-1 - periods] - 1) * 100


class GetQuoteTool(Tool):
    name = "get_quote"
    description = (
        "Get the latest price snapshot for a stock: current price, day change, 52-week "
        "range, volume, and market cap. Use this first to anchor any analysis in the "
        "current price."
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
        info = provider_call(self.provider.get_profile, t)
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev = info.get("previousClose") or info.get("regularMarketPreviousClose")
        return {
            "ticker": t,
            "name": info.get("longName") or info.get("shortName"),
            "currency": info.get("currency"),
            "price": price,
            "previous_close": prev,
            "day_change_pct": (price / prev - 1) * 100 if price and prev else None,
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "avg_volume_3m": info.get("averageVolume"),
            "market_cap": info.get("marketCap"),
        }


class GetPriceHistoryTool(Tool):
    name = "get_price_history"
    description = (
        "Get a compact summary of a stock's price history over a period: returns over "
        "several horizons, volatility, max drawdown, and roughly 25 evenly spaced closing "
        "prices to show the trend. Does NOT return every daily bar."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "ticker": TICKER_PROP,
            "period": {"type": "string", "enum": PERIODS, "description": "Look-back window"},
        },
        "required": ["ticker", "period"],
        "additionalProperties": False,
    }

    def __init__(self, provider: MarketDataProvider):
        self.provider = provider

    def run(self, ticker: str, period: str) -> dict[str, Any]:
        t = clean_ticker(ticker)
        df = provider_call(self.provider.get_price_history, t, period)
        close = df["Close"]
        daily = close.pct_change().dropna()
        drawdown = close / close.cummax() - 1
        step = max(1, len(close) // 25)
        sampled = close.iloc[::step]
        return {
            "ticker": t,
            "period": period,
            "start": close.index[0],
            "end": close.index[-1],
            "first_close": close.iloc[0],
            "last_close": close.iloc[-1],
            "total_return_pct": (close.iloc[-1] / close.iloc[0] - 1) * 100,
            "return_1m_pct": pct_change(close, 21),
            "return_3m_pct": pct_change(close, 63),
            "annualized_volatility_pct": float(daily.std() * (252 ** 0.5) * 100) if len(daily) > 1 else None,
            "max_drawdown_pct": float(drawdown.min() * 100),
            "high": close.max(),
            "low": close.min(),
            "sampled_closes": {d.date().isoformat(): round(float(v), 2) for d, v in sampled.items()},
        }


class GetMarketOverviewTool(Tool):
    name = "get_market_overview"
    description = (
        "Get a snapshot of the overall market: recent returns for major index ETFs "
        "(S&P 500, Nasdaq-100, Dow, Russell 2000), the VIX volatility index, the 10-year "
        "Treasury yield, gold, and the dollar. Use for market-outlook questions or to "
        "compare a stock against the market."
    )
    input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    def __init__(self, provider: MarketDataProvider):
        self.provider = provider

    def run(self) -> dict[str, Any]:
        out = {}
        for symbol, label in MARKET_GAUGES.items():
            try:
                close = self.provider.get_price_history(symbol, "1y")["Close"]
            except Exception as e:  # one missing gauge shouldn't sink the whole overview
                out[symbol] = {"label": label, "error": str(e)}
                continue
            out[symbol] = {
                "label": label,
                "last": close.iloc[-1],
                "change_1d_pct": pct_change(close, 1),
                "change_1m_pct": pct_change(close, 21),
                "change_3m_pct": pct_change(close, 63),
                "change_1y_pct": pct_change(close, len(close) - 1),
                "pct_from_52w_high": float((close.iloc[-1] / close.max() - 1) * 100),
            }
        return out
