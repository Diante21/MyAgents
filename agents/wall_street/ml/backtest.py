"""Backtester: would trading on the model's out-of-sample signals have made money?

Accuracy and AUC are statistics; a backtest asks the question that matters. Strategy:
  * For each stock, each day: hold it if prob > threshold, otherwise sit in cash.
  * Signal from day t's close is applied to the return from t to t+1 (no peeking).
  * Every switch in or out pays `cost_bps` (commissions + bid/ask spread + slippage).
  * The portfolio is an equal-weighted average of the per-stock sleeves.
Benchmark: buy-and-hold the same stocks over the same out-of-sample days.

A second strategy fits RANKING models better (run_top_bucket): every `horizon` days, buy
the top 20% of stocks by score in equal weights, hold them until the next rebalance, and
compare with holding all stocks equally. It is always fully invested, so it isolates
stock-SELECTION skill from market timing - the threshold strategy mixes both.

Caveats to remember: survivorship bias in the universe, no taxes, perfect fills at the
close. Real results would be worse than this.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestReport:
    strategy: dict
    benchmark: dict
    days: int
    start: str
    end: str


def performance(daily: pd.Series) -> dict:
    equity = (1 + daily).cumprod()
    years = len(daily) / 252
    total = equity.iloc[-1] - 1
    vol = daily.std() * np.sqrt(252)
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0.0
    return {
        "total_return_pct": round(total * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "sharpe": round(daily.mean() * 252 / vol, 2) if vol > 0 else 0.0,
        "max_drawdown_pct": round(float((equity / equity.cummax() - 1).min()) * 100, 2),
    }


class Backtester:
    def __init__(self, threshold: float = 0.5, cost_bps: float = 5.0):
        self.threshold = threshold
        self.cost = cost_bps / 10_000

    def run(self, oos: pd.DataFrame) -> BacktestReport:
        df = oos.sort_values(["ticker", "date"]).copy()
        df["position"] = (df["prob"] > self.threshold).astype(float)
        df["trade"] = df.groupby("ticker")["position"].diff().abs().fillna(df["position"])
        df["strat_ret"] = df["position"] * df["next_day_return"] - df["trade"] * self.cost

        strat = df.groupby("date")["strat_ret"].mean()
        bench = df.groupby("date")["next_day_return"].mean()

        s = performance(strat)
        s["exposure_pct"] = round(df["position"].mean() * 100, 1)       # time invested
        s["trades_per_stock_per_year"] = round(
            df["trade"].sum() / df["ticker"].nunique() / (len(strat) / 252), 1)
        return BacktestReport(
            strategy=s,
            benchmark=performance(bench),
            days=len(strat),
            start=str(pd.Timestamp(strat.index.min()).date()),
            end=str(pd.Timestamp(strat.index.max()).date()),
        )

    def run_top_bucket(self, oos: pd.DataFrame, rebalance_every: int, top_frac: float = 0.2) -> BacktestReport:
        """Hold the top `top_frac` of stocks by prob, rebalanced every `rebalance_every`
        trading days. Costs are charged on turnover (the fraction of the book replaced)."""
        df = oos.dropna(subset=["prob", "next_day_return"])
        dates = np.sort(df["date"].unique())
        rebalance_dates = set(dates[::max(rebalance_every, 1)])
        by_date = {d: g for d, g in df.groupby("date")}

        holdings: set[str] = set()
        daily, turnovers = [], []
        for d in dates:
            day = by_date[d]
            cost = 0.0
            if d in rebalance_dates:
                k = max(1, int(round(len(day) * top_frac)))
                new = set(day.nlargest(k, "prob")["ticker"])
                # One-way turnover of an equal-weight book: share of names replaced.
                turnover = 1.0 if not holdings else len(new - holdings) / len(new)
                cost = 2 * turnover * self.cost          # sell the old + buy the new
                turnovers.append(turnover)
                holdings = new
            # Signal from day d's close earns d -> d+1, like the threshold strategy.
            held = day[day["ticker"].isin(holdings)]["next_day_return"]
            daily.append((d, (held.mean() if len(held) else 0.0) - cost))

        strat = pd.Series(dict(daily)).sort_index()
        bench = df.groupby("date")["next_day_return"].mean()
        s = performance(strat)
        s["rebalance_every_days"] = rebalance_every
        s["avg_turnover_pct"] = round(float(np.mean(turnovers)) * 100, 1) if turnovers else 0.0
        return BacktestReport(
            strategy=s,
            benchmark=performance(bench),
            days=len(strat),
            start=str(pd.Timestamp(strat.index.min()).date()),
            end=str(pd.Timestamp(strat.index.max()).date()),
        )
