"""Settings shared by every agent, loaded once from environment variables / .env."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reads .env into os.environ. Existing environment variables win, so a key set
# in your shell or PyCharm run configuration overrides the file.
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # The anthropic SDK reads ANTHROPIC_API_KEY from the environment by itself,
    # so we never pass the key around in code. We only check it's present.
    model: str = os.getenv("AGENT_MODEL", "claude-opus-5-5")
    # effort controls how hard the model thinks: low | medium | high | xhigh | max.
    # On claude-opus-5-5 the API default is "medium"; research benefits from "high".
    effort: str = os.getenv("AGENT_EFFORT", "high")
    max_tokens: int = int(os.getenv("AGENT_MAX_TOKENS", "32000"))
    # Safety cap on the agent loop so a confused agent can't call tools forever.
    max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "20"))
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"   # trained ML models (gitignored)
    # Your holdings: a Robinhood activity report or a ticker,shares,avg_cost CSV.
    # Lives in data/ by default so it's gitignored - it's personal financial data.
    # `or` (not a getenv default) so an EMPTY value like PORTFOLIO_FILE="" also falls back -
    # Path("") would otherwise mean "the current folder".
    portfolio_file: Path = Path(os.getenv("PORTFOLIO_FILE") or PROJECT_ROOT / "data" / "portfolio.csv")

    @property
    def has_api_key(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))


settings = Settings()
