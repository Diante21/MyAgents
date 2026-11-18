"""Recent news headlines for a ticker."""

from typing import Any

from core.tools import Tool
from agents.wall_street.data.providers import MarketDataProvider
from agents.wall_street.tools import clean_ticker, provider_call
from agents.wall_street.tools.market_data import TICKER_PROP


class GetNewsTool(Tool):
    name = "get_news"
    description = (
        "Get up to 10 recent news headlines (with publisher, date, short summary, and URL) "
        "about a company. Use to find catalysts, risks, and sentiment. Headlines are not "
        "verified facts - weigh the source."
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
        items = provider_call(self.provider.get_news, t)[:10]
        return {"ticker": t, "count": len(items), "articles": items}
