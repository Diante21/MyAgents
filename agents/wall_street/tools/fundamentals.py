"""Company fundamentals: profile, valuation, profitability, statements, peer comparison."""

from typing import Any

from core.tools import Tool, ToolError
from agents.wall_street.data.providers import MarketDataProvider
from agents.wall_street.tools import clean_ticker, provider_call
from agents.wall_street.tools.market_data import TICKER_PROP

# yfinance `info` keys we care about -> friendlier names for Claude.
KEY_METRICS = {
    "marketCap": "market_cap",
    "enterpriseValue": "enterprise_value",
    "trailingPE": "pe_trailing",
    "forwardPE": "pe_forward",
    "pegRatio": "peg_ratio",
    "priceToSalesTrailing12Months": "price_to_sales",
    "priceToBook": "price_to_book",
    "enterpriseToEbitda": "ev_to_ebitda",
    "revenueGrowth": "revenue_growth_yoy",
    "earningsGrowth": "earnings_growth_yoy",
    "grossMargins": "gross_margin",
    "operatingMargins": "operating_margin",
    "profitMargins": "net_margin",
    "returnOnEquity": "return_on_equity",
    "freeCashflow": "free_cash_flow",
    "totalCash": "total_cash",
    "totalDebt": "total_debt",
    "debtToEquity": "debt_to_equity",
    "currentRatio": "current_ratio",
    "dividendYield": "dividend_yield",
    "payoutRatio": "payout_ratio",
    "beta": "beta",
    "heldPercentInstitutions": "institutional_ownership",
    "shortPercentOfFloat": "short_pct_of_float",
    "recommendationKey": "analyst_consensus",
    "targetMeanPrice": "analyst_target_mean",
    "numberOfAnalystOpinions": "analyst_count",
}

STATEMENT_LINES = {
    "income": ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "Diluted EPS"],
    "balance": ["Total Assets", "Total Debt", "Stockholders Equity", "Cash And Cash Equivalents"],
    "cashflow": ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow", "Repurchase Of Capital Stock"],
}


def extract_metrics(info: dict[str, Any]) -> dict[str, Any]:
    return {friendly: info.get(key) for key, friendly in KEY_METRICS.items()}


class GetFundamentalsTool(Tool):
    name = "get_fundamentals"
    description = (
        "Get a company's profile and key fundamentals: sector, industry, business "
        "description, valuation ratios (P/E, P/S, EV/EBITDA, PEG), growth, margins, "
        "return on equity, cash/debt, dividend, beta, and Wall Street analyst consensus. "
        "Ratios are decimals (0.25 = 25%)."
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
        return {
            "ticker": t,
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "employees": info.get("fullTimeEmployees"),
            "business_summary": (info.get("longBusinessSummary") or "")[:1200],
            "metrics": extract_metrics(info),
        }


class GetFinancialStatementsTool(Tool):
    name = "get_financial_statements"
    description = (
        "Get key lines from a company's annual income statement, balance sheet, and cash "
        "flow statement for the last few fiscal years (revenue, profit, EPS, debt, equity, "
        "free cash flow, buybacks). Use to judge growth trends and financial health."
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
        statements = provider_call(self.provider.get_financial_statements, t)
        out: dict[str, Any] = {"ticker": t}
        for kind, lines in STATEMENT_LINES.items():
            df = statements.get(kind)
            if df is None or df.empty:
                continue
            table = {}
            for line in lines:
                if line in df.index:
                    row = df.loc[line]
                    table[line] = {col.date().isoformat(): row[col] for col in df.columns}
            out[kind] = table
        return out


class ComparePeersTool(Tool):
    name = "compare_peers"
    description = (
        "Compare valuation, growth, and profitability metrics side by side for 2-8 tickers "
        "(e.g. a company and its competitors). Use to judge whether a stock is cheap or "
        "expensive relative to its peers."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tickers": {"type": "array", "items": TICKER_PROP, "description": "2 to 8 ticker symbols"},
        },
        "required": ["tickers"],
        "additionalProperties": False,
    }
    FIELDS = ["market_cap", "pe_trailing", "pe_forward", "price_to_sales", "ev_to_ebitda",
              "revenue_growth_yoy", "operating_margin", "net_margin", "return_on_equity",
              "debt_to_equity", "beta"]

    def __init__(self, provider: MarketDataProvider):
        self.provider = provider

    def run(self, tickers: list[str]) -> dict[str, Any]:
        if not 2 <= len(tickers) <= 8:
            raise ToolError("Provide between 2 and 8 tickers.")
        table = {}
        for raw in tickers:
            t = clean_ticker(raw)
            try:
                metrics = extract_metrics(self.provider.get_profile(t))
                table[t] = {f: metrics.get(f) for f in self.FIELDS}
            except Exception as e:
                table[t] = {"error": str(e)}
        return table
