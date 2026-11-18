"""Structured outputs for WallStreetAgent.

Passing one of these to BaseAgent.run(output_model=...) makes the API constrain the
final answer to JSON matching the schema, which we then validate into the model.
Benefits: no brittle text parsing, and results can be stored/compared/tested.

Keep schemas simple (no numeric min/max constraints) - structured outputs support a
subset of JSON Schema. Field descriptions are sent to the model, so write them as
instructions.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Metric(BaseModel):
    # A list of {name, value} instead of dict[str, str]: structured outputs need every
    # object to have fixed, known keys, so free-form dicts aren't allowed.
    name: str = Field(description="e.g. 'Forward P/E'")
    value: str = Field(description="e.g. '28.4' or '12.5%'")


class CompanyReport(BaseModel):
    ticker: str
    company_name: str
    as_of: str = Field(description="Date of the data used, YYYY-MM-DD")
    summary: str = Field(description="3-5 sentence executive summary of the investment picture")
    business_model: str = Field(description="How the company makes money and its competitive position")
    financial_health: str = Field(description="Growth, margins, cash flow, debt - with figures from tools")
    valuation: str = Field(description="Valuation vs. history and peers - with the actual multiples")
    technical_picture: str = Field(description="Trend, momentum, and where price sits vs. moving averages")
    recent_news: str = Field(description="What recent headlines suggest; say so if there is no notable news")
    bull_case: list[str] = Field(description="Strongest reasons the stock could outperform")
    bear_case: list[str] = Field(description="Strongest reasons the stock could underperform")
    key_metrics: list[Metric] = Field(description="5-10 headline metrics")
    data_gaps: list[str] = Field(description="Data that was missing or unreliable; empty if none")


class StockRecommendation(BaseModel):
    ticker: str
    action: Literal["buy", "hold", "sell", "avoid"]
    confidence: Literal["low", "medium", "high"]
    time_horizon: Literal["short (< 3 months)", "medium (3-12 months)", "long (> 1 year)"]
    thesis: str = Field(description="Why, in 2-4 sentences, citing specific figures")
    model_signal: str = Field(description=(
        "What the predict_price_move model said (probability + its track record) and how much "
        "weight you gave it; 'not available' if the tool wasn't offered"))
    key_risks: list[str]
    what_would_change_my_mind: str = Field(description="Concrete signal that would flip the view")


class RecommendationSet(BaseModel):
    risk_profile: str
    recommendations: list[StockRecommendation] = Field(description="One per ticker, best idea first")
    portfolio_notes: str = Field(description=(
        "Diversification/concentration notes across these picks - and, if the get_portfolio "
        "tool is available, how each pick would fit with what the user already owns"))


class HoldingReview(BaseModel):
    ticker: str
    weight_pct: str = Field(description="Share of the total portfolio, e.g. '23.4%'")
    unrealized_gain: str = Field(description="e.g. '+$1,240 (+31.2%)' or 'unknown cost basis'")
    assessment: str = Field(description="1-3 sentences: role in the portfolio, valuation/momentum, any concern")
    suggested_stance: Literal["keep", "trim", "add", "review closely"]


class PortfolioReview(BaseModel):
    as_of: str = Field(description="Date of the price data used, YYYY-MM-DD")
    total_value: str
    summary: str = Field(description="3-5 sentence overview: what the portfolio is, how it's doing, the main issue")
    holdings: list[HoldingReview] = Field(description="Every position, largest first")
    diversification: str = Field(description="Position and sector concentration, correlation, effective holdings")
    risk: str = Field(description="Beta, volatility and drawdown vs. the S&P 500, in plain language")
    strengths: list[str]
    concerns: list[str]
    ideas_to_consider: list[str] = Field(description=(
        "Concrete ideas for the user to consider themselves (e.g. 'tech is 68% - consider whether '"
        "that matches your risk tolerance'). These are suggestions, not orders"))
    data_warnings: list[str] = Field(description="Issues with the portfolio data the user should check; empty if none")
