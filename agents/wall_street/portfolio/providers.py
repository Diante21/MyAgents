"""Where holdings come from. Same pattern as MarketDataProvider: tools depend on the
abstract PortfolioProvider, so a future source (Plaid, an official Robinhood API, another
broker) is a new class here - no tool or agent code changes.

Two sources today:

1. HoldingsCsvProvider - a simple file you maintain yourself:
       ticker,shares,avg_cost
       AAPL,10,150.25
       NVDA,5.5,120
       CASH,1200          <- optional cash row (amount in the shares column)

2. RobinhoodActivityCsvProvider - Robinhood's own export. Robinhood doesn't export a
   positions list, but it does export an account activity report (Account -> Reports &
   statements -> Reports -> generate a report covering ALL dates since you opened the
   account). We rebuild holdings by replaying every buy, sell and split in date order.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import csv
import re

from agents.wall_street.portfolio.models import Portfolio, Position


class PortfolioError(Exception):
    """The portfolio file is missing or not in a format we understand."""


class PortfolioProvider(ABC):
    @abstractmethod
    def get_portfolio(self) -> Portfolio: ...


# ---- 1. simple holdings file ----------------------------------------------------------
class HoldingsCsvProvider(PortfolioProvider):
    def __init__(self, path: Path):
        self.path = path

    def get_portfolio(self) -> Portfolio:
        rows = _read_csv(self.path)
        positions, cash, warnings = [], None, []
        for i, row in enumerate(rows, start=2):           # line 1 is the header
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            shares = _number(row.get("shares"))
            if shares is None:
                warnings.append(f"Line {i}: no share count for {ticker}, skipped")
                continue
            if ticker == "CASH":
                cash = shares
                continue
            positions.append(Position(ticker, shares, _number(row.get("avg_cost"))))
        return Portfolio(positions, cash=cash, source=self.path.name, warnings=warnings)


# ---- 2. Robinhood activity report -----------------------------------------------------
class RobinhoodActivityCsvProvider(PortfolioProvider):
    # Transaction codes that change the share count, and ones we can't replay safely.
    BUY, SELL, SPLIT = {"Buy"}, {"Sell"}, {"SPL"}
    SHARE_CHANGING_UNSUPPORTED = {"SXCH", "MRGS", "MRGC", "REC", "ACATI", "ACATO", "SOFF", "CONV"}

    def __init__(self, path: Path):
        self.path = path

    def get_portfolio(self) -> Portfolio:
        rows = [r for r in _read_csv(self.path) if (r.get("Trans Code") or "").strip()]
        if not rows:
            raise PortfolioError(f"{self.path.name} has no transactions.")
        # Robinhood lists newest first; replay oldest first.
        rows.sort(key=lambda r: _date_key(r.get("Activity Date", "")))

        shares: dict[str, float] = {}
        cost: dict[str, float] = {}
        cash = 0.0
        warnings: list[str] = []

        for r in rows:
            code = r["Trans Code"].strip()
            ticker = (r.get("Instrument") or "").strip().upper()
            qty = _number(r.get("Quantity")) or 0.0
            price = _number(r.get("Price")) or 0.0
            cash += _number(r.get("Amount")) or 0.0       # every row's cash effect

            if code in self.BUY and ticker:
                shares[ticker] = shares.get(ticker, 0.0) + qty
                cost[ticker] = cost.get(ticker, 0.0) + qty * price
            elif code in self.SELL and ticker:
                held = shares.get(ticker, 0.0)
                if held > 0:
                    # Average-cost method: selling removes cost at the current average.
                    cost[ticker] -= cost[ticker] / held * min(qty, held)
                shares[ticker] = held - qty
            elif code in self.SPLIT and ticker:
                shares[ticker] = shares.get(ticker, 0.0) + qty   # extra shares, same total cost
            elif code in self.SHARE_CHANGING_UNSUPPORTED:
                warnings.append(f"{r.get('Activity Date')}: {code} for {ticker or '?'} "
                                f"({(r.get('Description') or '').strip()[:60]}) not replayed - "
                                "check that position by hand")

        positions = []
        for ticker, qty in sorted(shares.items()):
            if qty < -1e-6:
                warnings.append(f"{ticker}: replay gives {qty:.4f} shares - the report probably "
                                "doesn't start at account opening")
            if qty > 1e-6:
                positions.append(Position(ticker, round(qty, 6), round(cost[ticker] / qty, 4)))

        newest = max((r.get("Activity Date", "") for r in rows), key=_date_key)
        warnings.append("Cash is estimated by summing every transaction amount in the report.")
        return Portfolio(positions, cash=round(cash, 2), source=self.path.name,
                         as_of=newest, warnings=warnings)


# ---- choosing a reader ----------------------------------------------------------------
def provider_for(path: Path) -> PortfolioProvider:
    """Pick the right reader from the file's header row."""
    if not path.exists():
        raise PortfolioError(f"Portfolio file not found: {path}")
    header = path.read_text(encoding="utf-8-sig").splitlines()[0].lower()
    if "trans code" in header:
        return RobinhoodActivityCsvProvider(path)
    if "ticker" in header and "shares" in header:
        return HoldingsCsvProvider(path)
    raise PortfolioError(
        f"Don't recognise {path.name}. Use a Robinhood activity report, or a CSV with "
        "columns ticker,shares,avg_cost (see docs/portfolio.example.csv).")


# ---- helpers --------------------------------------------------------------------------
def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise PortfolioError(f"Portfolio file not found: {path}")
    # utf-8-sig strips the invisible BOM Excel adds; Robinhood rows can span lines (csv handles it).
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{(k or "").strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def _number(value: str | None) -> float | None:
    """'$1,234.56' -> 1234.56, '($12.00)' -> -12.0, '10S' -> 10.0, '' -> None."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^0-9.\-]", "", text)
    if cleaned in ("", "-", "."):
        return None
    number = float(cleaned)
    return -abs(number) if negative else number


def _date_key(value: str) -> tuple[int, int, int]:
    """'3/7/2025' -> (2025, 3, 7) for sorting. Unparseable dates sort first."""
    try:
        month, day, year = (int(x) for x in value.strip().split("/"))
        return (year, month, day)
    except ValueError:
        return (0, 0, 0)
