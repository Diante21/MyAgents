"""WallStreetAgent tools. Each is a thin wrapper: validate input -> call a provider or
pure function -> return a small JSON-able dict for Claude to read."""

import re

from core.tools import ToolError
from agents.wall_street.data.providers import DataUnavailable

_TICKER = re.compile(r"^[A-Z0-9.\-^=]{1,12}$")


def clean_ticker(ticker: str) -> str:
    t = ticker.strip().upper()
    if not _TICKER.match(t):
        raise ToolError(f"{ticker!r} doesn't look like a ticker symbol (e.g. AAPL, BRK-B, ^GSPC).")
    return t


def provider_call(fn, *args, **kwargs):
    """Turn provider 'no data' errors into ToolErrors Claude can read and react to."""
    try:
        return fn(*args, **kwargs)
    except DataUnavailable as e:
        raise ToolError(str(e)) from e
