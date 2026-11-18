import json

import numpy as np
import pandas as pd
import pytest

from core.schema_utils import to_strict_json_schema
from core.tools import ToolError, to_json_text
from agents.wall_street import WallStreetAgent
from agents.wall_street.schemas import CompanyReport, RecommendationSet
from agents.wall_street.tools import clean_ticker
from agents.wall_street.tools.technicals import compute_indicators, rsi
from tests.conftest import FakeLLM


def test_all_tool_schemas_are_strict(provider):
    agent = WallStreetAgent(llm=FakeLLM([]), provider=provider)
    for tool in agent.tools.api_definitions():
        schema = tool["input_schema"]
        assert schema["additionalProperties"] is False, tool["name"]
        assert set(schema["required"]) == set(schema["properties"]), tool["name"]
        assert len(tool["description"]) > 40, f"{tool['name']} needs a real description"


def test_every_tool_runs_on_fake_data(provider):
    agent = WallStreetAgent(llm=FakeLLM([]), provider=provider)
    inputs = {
        "get_quote": {"ticker": "AAPL"},
        "get_price_history": {"ticker": "AAPL", "period": "1y"},
        "get_market_overview": {},
        "get_fundamentals": {"ticker": "AAPL"},
        "get_financial_statements": {"ticker": "AAPL"},
        "compare_peers": {"tickers": ["AAPL", "MSFT"]},
        "compute_technicals": {"ticker": "AAPL"},
        "get_news": {"ticker": "AAPL"},
    }
    assert set(inputs) == set(agent.tools.names)
    for name, tool_input in inputs.items():
        result = agent.tools.execute(name, tool_input, "id")
        assert not result.get("is_error"), (name, result["content"])
        json.loads(result["content"])  # valid JSON


def test_structured_schemas_are_closed():
    for model in (CompanyReport, RecommendationSet):
        text = json.dumps(to_strict_json_schema(model))
        assert "$ref" not in text

        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    assert node.get("additionalProperties") is False, node
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(to_strict_json_schema(model))


def test_clean_ticker():
    assert clean_ticker(" brk-b ") == "BRK-B"
    with pytest.raises(ToolError):
        clean_ticker("not a ticker!")


def test_rsi_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(up).iloc[-1] > 99
    assert rsi(up[::-1].reset_index(drop=True)).iloc[-1] < 1


def test_compute_indicators_on_uptrend(provider):
    out = compute_indicators(provider.df)
    assert out["sma_50"] > out["sma_200"]
    assert out["trend_regime"].startswith("golden")
    assert out["pct_vs_sma_20"] > 0


def test_to_json_text_handles_nan_and_numpy():
    text = to_json_text({"a": float("nan"), "b": np.float64(1.23456789), "c": pd.Timestamp("2026-01-02")})
    assert json.loads(text) == {"a": None, "b": 1.2346, "c": "2026-01-02"}
