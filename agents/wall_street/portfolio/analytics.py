"""Portfolio math as pure functions (no network), so it's easy to test and reuse.

Concepts worth knowing:
  * Weight: a position's share of the portfolio's value.
  * HHI (Herfindahl-Hirschman index): sum of squared weights. 1/HHI is the "effective
    number of holdings": 10 equal positions -> 10; one 90% position + 9 tiny -> ~1.2.
    It says how diversified you REALLY are, regardless of how many tickers you own.
  * Beta: how much the portfolio moves when the market moves 1%. 1.3 = 30% more swing.
  * Correlation: owning 10 chip stocks isn't diversification if they all move together.
"""

from typing import Any

import numpy as np
import pandas as pd

from agents.wall_street.portfolio.models import Portfolio


def valuate(portfolio: Portfolio, prices: dict[str, float], sectors: dict[str, str]) -> dict[str, Any]:
    """Market value, weight and gain/loss per position, plus totals."""
    rows = []
    for p in portfolio.positions:
        price = prices.get(p.ticker)
        value = p.shares * price if price is not None else None
        cost = p.cost_basis
        rows.append({
            "ticker": p.ticker,
            "sector": sectors.get(p.ticker, "Unknown"),
            "shares": p.shares,
            "avg_cost": p.avg_cost,
            "price": price,
            "market_value": value,
            "cost_basis": cost,
            "unrealized_gain": value - cost if value is not None and cost is not None else None,
            "unrealized_gain_pct": (value / cost - 1) * 100 if value is not None and cost else None,
        })

    invested = sum(r["market_value"] or 0 for r in rows)
    cash = portfolio.cash or 0.0
    total = invested + cash
    for r in rows:
        r["weight_pct"] = (r["market_value"] / total * 100) if r["market_value"] and total else None
    rows.sort(key=lambda r: r["market_value"] or 0, reverse=True)

    known_cost = [r for r in rows if r["cost_basis"] is not None and r["market_value"] is not None]
    total_cost = sum(r["cost_basis"] for r in known_cost)
    total_gain = sum(r["unrealized_gain"] for r in known_cost)
    return {
        "positions": rows,
        "invested_value": invested,
        "cash": portfolio.cash,
        "total_value": total,
        "cash_pct": cash / total * 100 if total else None,
        "total_unrealized_gain": total_gain if known_cost else None,
        "total_unrealized_gain_pct": total_gain / total_cost * 100 if total_cost else None,
        "missing_prices": [r["ticker"] for r in rows if r["price"] is None],
    }


def concentration(valuation: dict[str, Any]) -> dict[str, Any]:
    """How diversified the portfolio really is - by position and by sector."""
    rows = [r for r in valuation["positions"] if r["market_value"]]
    invested = sum(r["market_value"] for r in rows)
    if not invested:
        return {}
    weights = np.array([r["market_value"] / invested for r in rows])   # of invested money
    sector_weights: dict[str, float] = {}
    for r in rows:
        sector_weights[r["sector"]] = sector_weights.get(r["sector"], 0) + r["market_value"] / invested * 100
    hhi = float((weights ** 2).sum())
    return {
        "number_of_positions": len(rows),
        "effective_number_of_holdings": round(1 / hhi, 1),
        "largest_position": rows[0]["ticker"],
        "largest_position_pct_of_invested": round(weights[0] * 100, 1),
        "top_3_pct_of_invested": round(weights[:3].sum() * 100, 1),
        "sector_pct_of_invested": dict(sorted(((k, round(v, 1)) for k, v in sector_weights.items()),
                                              key=lambda kv: -kv[1])),
    }


def risk_metrics(weights: dict[str, float], returns: pd.DataFrame, market: str = "SPY") -> dict[str, Any]:
    """weights: ticker -> weight of invested money (sums to 1).
    returns: daily returns, one column per ticker plus the market column.
    Uses today's weights over the past year ("what if I'd held this all year")."""
    tickers = [t for t in weights if t in returns.columns]
    if not tickers or market not in returns.columns:
        return {}
    data = returns[tickers + [market]].dropna()
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    port = data[tickers].to_numpy() @ w
    mkt = data[market].to_numpy()

    def beta(series: np.ndarray) -> float:
        return float(np.cov(series, mkt)[0, 1] / np.var(mkt, ddof=1))

    equity = np.cumprod(1 + port)
    corr = data[tickers].corr().to_numpy()
    off_diag = corr[~np.eye(len(tickers), dtype=bool)]
    return {
        "days_used": len(data),
        "portfolio_beta": round(beta(port), 2),
        "portfolio_volatility_pct": round(float(port.std(ddof=1) * np.sqrt(252) * 100), 1),
        "market_volatility_pct": round(float(mkt.std(ddof=1) * np.sqrt(252) * 100), 1),
        "hypothetical_1y_return_pct": round(float(equity[-1] - 1) * 100, 1),
        "market_1y_return_pct": round(float(np.prod(1 + mkt) - 1) * 100, 1),
        "hypothetical_max_drawdown_pct": round(float((equity / np.maximum.accumulate(equity) - 1).min()) * 100, 1),
        "average_pairwise_correlation": round(float(off_diag.mean()), 2) if len(off_diag) else None,
        "beta_by_position": {t: round(beta(data[t].to_numpy()), 2) for t in tickers},
    }
