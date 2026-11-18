import numpy as np
import pandas as pd
import pytest

from agents.wall_street import WallStreetAgent
from agents.wall_street.portfolio.analytics import concentration, risk_metrics, valuate
from agents.wall_street.portfolio.models import Portfolio, Position
from agents.wall_street.portfolio.providers import (
    HoldingsCsvProvider, PortfolioError, RobinhoodActivityCsvProvider, _number, provider_for,
)
from agents.wall_street.tools.portfolio import AnalyzePortfolioRiskTool, GetPortfolioTool
from tests.conftest import FakeLLM, FakeProvider

# Shaped like Robinhood's activity report: newest first, $ and (negative) amounts,
# a multi-line description, and the disclaimer footer with no transaction code.
ROBINHOOD_CSV = '''"Activity Date","Process Date","Settle Date","Instrument","Description","Trans Code","Quantity","Price","Amount"
"6/2/2025","6/2/2025","6/3/2025","AAPL","Apple
CUSIP: 037833100","Sell","5","$200.00","$1,000.00"
"3/10/2025","3/10/2025","3/11/2025","NVDA","NVIDIA Stock Split","SPL","9","",""
"2/1/2025","2/1/2025","2/3/2025","NVDA","NVIDIA","Buy","1","$800.00","($800.00)"
"1/15/2025","1/15/2025","1/16/2025","AAPL","Apple","Buy","10","$150.00","($1,500.00)"
"1/10/2025","1/10/2025","1/10/2025","","ACH Deposit","ACH","","","$5,000.00"
"","","","","","","","","The data provided is for informational purposes only."
'''


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---- readers --------------------------------------------------------------------------
def test_robinhood_report_rebuilds_holdings(tmp_path):
    p = RobinhoodActivityCsvProvider(write(tmp_path, "rh.csv", ROBINHOOD_CSV)).get_portfolio()
    by = {pos.ticker: pos for pos in p.positions}
    assert by["AAPL"].shares == 5 and by["AAPL"].avg_cost == 150.0   # avg cost unchanged by a sale
    assert by["NVDA"].shares == 10 and by["NVDA"].avg_cost == 80.0   # 10-for-1 split: 1 -> 10 shares
    assert p.cash == 5000 - 1500 - 800 + 1000
    assert p.as_of == "6/2/2025"


def test_unsupported_codes_become_warnings(tmp_path):
    csv = ROBINHOOD_CSV.replace('"SPL"', '"SXCH"')
    p = RobinhoodActivityCsvProvider(write(tmp_path, "rh.csv", csv)).get_portfolio()
    assert any("SXCH" in w for w in p.warnings)


def test_holdings_csv_with_cash_row(tmp_path):
    path = write(tmp_path, "h.csv", "ticker,shares,avg_cost\naapl,10,150\nNVDA,2,\nCASH,500,\n")
    p = HoldingsCsvProvider(path).get_portfolio()
    assert [(x.ticker, x.shares, x.avg_cost) for x in p.positions] == [("AAPL", 10, 150), ("NVDA", 2, None)]
    assert p.cash == 500


def test_provider_detection(tmp_path):
    assert isinstance(provider_for(write(tmp_path, "a.csv", ROBINHOOD_CSV)), RobinhoodActivityCsvProvider)
    assert isinstance(provider_for(write(tmp_path, "b.csv", "ticker,shares\nAAPL,1\n")), HoldingsCsvProvider)
    with pytest.raises(PortfolioError):
        provider_for(write(tmp_path, "c.csv", "foo,bar\n1,2\n"))
    with pytest.raises(PortfolioError):
        provider_for(tmp_path / "missing.csv")


@pytest.mark.parametrize("text,expected", [
    ("$1,234.56", 1234.56), ("($12.00)", -12.0), ("10S", 10.0), ("", None), ("-3", -3.0)])
def test_number_parsing(text, expected):
    assert _number(text) == expected


# ---- analytics ------------------------------------------------------------------------
def test_valuation_and_concentration():
    portfolio = Portfolio([Position("AAA", 10, 50), Position("BBB", 10, None)], cash=500)
    v = valuate(portfolio, {"AAA": 100.0, "BBB": 50.0}, {"AAA": "Tech", "BBB": "Energy"})
    assert v["total_value"] == 2000
    aaa = v["positions"][0]
    assert aaa["ticker"] == "AAA" and aaa["unrealized_gain"] == 500 and aaa["weight_pct"] == 50
    assert v["total_unrealized_gain"] == 500      # BBB has no cost basis -> excluded
    c = concentration(v)
    assert c["largest_position_pct_of_invested"] == pytest.approx(66.7, abs=0.1)
    assert c["sector_pct_of_invested"]["Tech"] == pytest.approx(66.7, abs=0.1)


def test_risk_metrics_beta():
    rng = np.random.default_rng(0)
    mkt = rng.normal(0, 0.01, 250)
    returns = pd.DataFrame({"SPY": mkt, "LEVERED": 2 * mkt, "SAME": mkt})
    r = risk_metrics({"LEVERED": 0.5, "SAME": 0.5}, returns)
    assert r["beta_by_position"]["LEVERED"] == pytest.approx(2.0)
    assert r["portfolio_beta"] == pytest.approx(1.5)
    assert r["average_pairwise_correlation"] == pytest.approx(1.0)


# ---- tools + agent wiring ---------------------------------------------------------------
@pytest.fixture
def holdings(tmp_path):
    return HoldingsCsvProvider(write(tmp_path, "h.csv", "ticker,shares,avg_cost\nAAPL,10,100\nMSFT,5,200\nCASH,1000,\n"))


def test_portfolio_tools_run(holdings):
    out = GetPortfolioTool(holdings, FakeProvider()).run()
    assert out["total_value"] == 15 * 150 + 1000          # FakeProvider prices everything at 150
    risk = AnalyzePortfolioRiskTool(holdings, FakeProvider()).run()
    assert "portfolio_beta" in risk


def test_agent_has_portfolio_tools_but_no_trading_tools(holdings):
    agent = WallStreetAgent(llm=FakeLLM([]), provider=FakeProvider(), portfolio=holdings)
    assert {"get_portfolio", "analyze_portfolio_risk"} <= set(agent.tools.names)
    trading_words = ("order", "trade", "buy", "sell", "execute")
    assert not [n for n in agent.tools.names if any(w in n for w in trading_words)]


def test_agent_without_portfolio_file_has_no_portfolio_tools():
    # conftest points settings.portfolio_file at a file that doesn't exist
    agent = WallStreetAgent(llm=FakeLLM([]), provider=FakeProvider())
    assert "get_portfolio" not in agent.tools.names
