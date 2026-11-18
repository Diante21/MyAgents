"""Test doubles. No network, no API key, no cost.

FakeLLM replays scripted responses shaped like the SDK's Message objects. This is how
you test agent LOGIC (does the loop run tools, pass results back, stop correctly?)
without paying for or depending on a real model.
"""

from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.wall_street.data.providers import DataUnavailable, MarketDataProvider  # noqa: E402
from core.llm import UsageTracker  # noqa: E402


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, tool_input, block_id="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def response(content, stop_reason):
    usage = SimpleNamespace(input_tokens=100, output_tokens=50,
                            cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)


class FakeLLM:
    model = "claude-opus-5-5"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, system, messages, tools=None, output_schema=None):
        # Snapshot the messages: the agent keeps appending to the same list.
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools,
                           "output_schema": output_schema})
        return self.responses.pop(0)

    def new_usage_tracker(self):
        return UsageTracker(model=self.model)


class FakeProvider(MarketDataProvider):
    def __init__(self, days=300):
        idx = pd.bdate_range(end="2026-11-13", periods=days)
        close = pd.Series(np.linspace(100, 150, days), index=idx)
        self.df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                                "Close": close, "Volume": 1_000_000.0})

    def get_price_history(self, ticker, period="1y", interval="1d"):
        if ticker == "BAD":
            raise DataUnavailable("No price history for 'BAD'")
        return self.df

    def get_profile(self, ticker):
        if ticker == "BAD":
            raise DataUnavailable("No company data for 'BAD'")
        return {"longName": f"{ticker} Corp", "currentPrice": 150.0, "previousClose": 148.0,
                "trailingPE": 25.0, "sector": "Technology", "marketCap": 1e12}

    def get_financial_statements(self, ticker):
        cols = [pd.Timestamp("2025-12-31"), pd.Timestamp("2024-12-31")]
        income = pd.DataFrame({cols[0]: [1000.0, 200.0], cols[1]: [900.0, 150.0]},
                              index=["Total Revenue", "Net Income"])
        return {"income": income, "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}

    def get_news(self, ticker):
        return [{"title": "Headline", "publisher": "Wire", "published": "2026-11-12",
                 "summary": "", "url": "https://example.com"}]


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture(autouse=True)
def isolated_user_files(tmp_path):
    """Point the model registry and portfolio file at empty temp locations, so a model
    you trained (models/) or your real portfolio (data/portfolio.csv) never changes what
    the tests see."""
    from core.config import settings
    originals = {name: getattr(settings, name) for name in ("models_dir", "portfolio_file")}
    # Settings is a frozen dataclass; object.__setattr__ bypasses that for tests only.
    object.__setattr__(settings, "models_dir", tmp_path / "models")
    object.__setattr__(settings, "portfolio_file", tmp_path / "no_portfolio.csv")
    yield
    for name, value in originals.items():
        object.__setattr__(settings, name, value)
