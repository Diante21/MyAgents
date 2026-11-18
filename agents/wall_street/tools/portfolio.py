"""Read-only portfolio tools: what you own, what it's worth, how concentrated and risky it is.

Privacy note: like every tool result, the output of these tools is sent to the Claude API
as part of the conversation. Holdings never leave your machine any other way, and the
portfolio file itself lives in data/ (gitignored).
"""

from typing import Any

import pandas as pd

from core.tools import Tool, ToolError
from agents.wall_street.data.providers import MarketDataProvider
from agents.wall_street.portfolio.analytics import concentration, risk_metrics, valuate
from agents.wall_street.portfolio.providers import PortfolioError, PortfolioProvider

NO_INPUT = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


class _PortfolioTool(Tool):
    def __init__(self, portfolio_provider: PortfolioProvider, market: MarketDataProvider):
        self.portfolio_provider = portfolio_provider
        self.market = market

    def _load(self):
        try:
            portfolio = self.portfolio_provider.get_portfolio()
        except PortfolioError as e:
            raise ToolError(str(e)) from e
        if not portfolio.positions:
            raise ToolError("The portfolio file has no open positions.")
        return portfolio

    def _price_and_sector(self, ticker: str) -> tuple[float | None, str]:
        try:
            info = self.market.get_profile(ticker)
        except Exception:
            return None, "Unknown"
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if info.get("quoteType") == "ETF":
            sector = f"ETF ({info.get('category') or 'fund'})"
        else:
            sector = info.get("sector") or "Unknown"
        return price, sector


class GetPortfolioTool(_PortfolioTool):
    name = "get_portfolio"
    description = (
        "Get the user's current stock holdings (read-only): each position's shares, average "
        "cost, live price, market value, weight, and unrealized gain/loss, plus cash, totals, "
        "and concentration by position and sector. Call this before giving advice that "
        "depends on what the user already owns."
    )
    input_schema = NO_INPUT

    def run(self) -> dict[str, Any]:
        portfolio = self._load()
        prices, sectors = {}, {}
        for ticker in portfolio.tickers:
            prices[ticker], sectors[ticker] = self._price_and_sector(ticker)
        valuation = valuate(portfolio, prices, sectors)
        return {
            "source": portfolio.source,
            "data_as_of": portfolio.as_of,
            **valuation,
            "concentration": concentration(valuation),
            "data_warnings": portfolio.warnings,
        }


class AnalyzePortfolioRiskTool(_PortfolioTool):
    name = "analyze_portfolio_risk"
    description = (
        "Measure the risk of the user's current holdings using the past year of daily prices: "
        "portfolio beta and volatility vs. the S&P 500, a hypothetical max drawdown, average "
        "correlation between holdings (high = less real diversification), and each position's "
        "beta. Assumes today's weights were held all year."
    )
    input_schema = NO_INPUT

    def run(self) -> dict[str, Any]:
        portfolio = self._load()
        closes, values, skipped = {}, {}, []
        for p in portfolio.positions:
            try:
                close = self.market.get_price_history(p.ticker, "1y")["Close"]
            except Exception:
                skipped.append(p.ticker)
                continue
            closes[p.ticker] = close
            values[p.ticker] = p.shares * float(close.iloc[-1])
        try:
            closes["SPY"] = self.market.get_price_history("SPY", "1y")["Close"]
        except Exception as e:
            raise ToolError(f"Couldn't load S&P 500 prices for comparison: {e}") from e
        if not values:
            raise ToolError("No price history for any holding.")

        returns = pd.DataFrame(closes).pct_change().iloc[1:]
        total = sum(values.values())
        weights = {t: v / total for t, v in values.items()}
        result = risk_metrics(weights, returns)
        result["skipped_no_data"] = skipped
        return result
