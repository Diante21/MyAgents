"""Plain data objects for holdings."""

from dataclasses import dataclass, field


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float | None = None      # per share; None if unknown

    @property
    def cost_basis(self) -> float | None:
        return self.shares * self.avg_cost if self.avg_cost is not None else None


@dataclass
class Portfolio:
    positions: list[Position]
    cash: float | None = None          # None if the source doesn't say
    source: str = ""                   # where the data came from (file name)
    as_of: str | None = None           # date of the newest data in the source
    warnings: list[str] = field(default_factory=list)  # things the reader couldn't handle

    @property
    def tickers(self) -> list[str]:
        return [p.ticker for p in self.positions]
