# MyAgents — WallStreetAgent Roadmap & Class Design

## Core design decision

You can't "train Claude" on stock data. Through the `anthropic` library, Claude is a fixed model
you send requests to. So the work is split in two:

1. **The LLM (Claude) is the analyst and orchestrator.** It reasons, researches companies, reads
   news and filings, decides which tools to call, and writes the final recommendation.
2. **Your own ML model is the forecaster.** You train a regular model (gradient boosting, LSTM, ...)
   on stock datasets, and the agent calls it as a tool: `predict_price_move("NVDA", horizon=5)`
   returns a probability, and Claude weighs it against everything else.

Realistic expectation: daily stock direction is close to a coin flip. A model that reaches 52–55%
out-of-sample accuracy after trading costs is a good result. Backtest honestly. Treat the agent's
output as research to support decisions, not advice to follow blindly.

### Setup notes
- Use lowercase package names (`agents/wall_street/`, not `../agents`).
- The venv is Python 3.14. Before the ML phase, check that PyTorch / LightGBM have 3.14 Windows
  wheels; if not, create a 3.12 venv.

## Directory structure

```
MyAgents/
├── main.py                       # CLI entry: pick an agent, send it a task
├── pyproject.toml                # dependencies
├── .env                          # ANTHROPIC_API_KEY, data-provider keys (gitignored)
│
├── core/                         # shared by EVERY agent
│   ├── config.py                 # Settings (loads .env)
│   ├── llm.py                    # LLMClient: wraps the anthropic SDK
│   ├── tools.py                  # Tool base class + ToolRegistry
│   ├── memory.py                 # ConversationMemory, LongTermMemory
│   ├── messages.py               # Task, AgentResult, AgentMessage dataclasses
│   ├── base_agent.py             # BaseAgent (abstract): the agent loop lives here
│   └── orchestrator.py           # Orchestrator: routes tasks between agents (later phase)
│
├── agents/
│   └── wall_street/
│       ├── agent.py              # WallStreetAgent(BaseAgent)
│       ├── prompts.py            # system prompt(s)
│       ├── schemas.py            # CompanyReport, StockRecommendation, PricePrediction
│       ├── tools/
│       │   ├── market_data.py    # GetQuoteTool, GetPriceHistoryTool
│       │   ├── fundamentals.py   # GetFinancialsTool, GetFilingsTool (SEC EDGAR)
│       │   ├── news.py           # GetNewsTool
│       │   ├── technicals.py     # ComputeIndicatorsTool (RSI, MACD, moving averages)
│       │   ├── macro.py          # GetMacroDataTool (FRED: rates, CPI, unemployment)
│       │   └── predict.py        # PredictPriceMoveTool -> calls ml/
│       ├── data/
│       │   ├── providers.py      # MarketDataProvider (abstract) + implementations
│       │   └── cache.py          # DataCache (Parquet/SQLite)
│       └── ml/
│           ├── features.py       # FeatureBuilder
│           ├── dataset.py        # DatasetBuilder (labels, time-based splits)
│           ├── models.py         # BasePredictor + concrete models
│           ├── trainer.py        # Trainer (walk-forward validation)
│           ├── backtest.py       # Backtester + metrics
│           └── registry.py       # ModelRegistry (versioned models)
│
├── data/                         # raw/ and processed/ datasets (gitignored)
├── models/                       # trained model files (gitignored)
├── notebooks/                    # exploration
└── tests/
```

Key idea: **`core/` knows nothing about stocks.** Future agents reuse it and only add their own
`agents/<name>/` folder.

## Class structure

### `core/`

```python
# core/messages.py
@dataclass
class Task:
    id: str
    objective: str                 # "Research NVDA and give a buy/hold/sell view"
    context: dict                  # extra inputs (portfolio, risk tolerance, ...)
    requested_by: str              # "user" or another agent's name

@dataclass
class AgentResult:
    task_id: str
    agent: str
    output: str | BaseModel        # text or a structured schema object
    tool_calls: list[dict]         # audit trail
    usage: dict                    # tokens / cost

@dataclass
class AgentMessage:                # agent-to-agent (later phase)
    sender: str
    recipient: str
    task: Task
```

```python
# core/llm.py
class LLMClient:
    """The only place that imports `anthropic`."""
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high"): ...
    def chat(self, system: str, messages: list, tools: list[dict]) -> Message: ...
    def stream(self, system: str, messages: list, tools: list[dict]) -> Message: ...
    def structured(self, system: str, messages: list, schema: type[BaseModel]) -> BaseModel: ...
    # handles retries, typed errors, usage tracking, stop_reason checks (incl. "refusal")
```

```python
# core/tools.py
class Tool(ABC):
    name: str
    description: str               # Claude reads this to decide when to call it
    input_schema: dict             # JSON schema (or derived from a Pydantic model)

    @abstractmethod
    def run(self, **kwargs) -> str: ...      # JSON/text for Claude to read
    def to_api_dict(self) -> dict: ...       # {"name", "description", "input_schema", "strict": True}

class ToolRegistry:
    def register(self, tool: Tool) -> None: ...
    def get(self, name: str) -> Tool: ...
    def api_definitions(self) -> list[dict]: ...
    def execute(self, name: str, tool_input: dict) -> dict: ...  # tool_result block; is_error on failure
```

```python
# core/memory.py
class ConversationMemory:          # short-term: messages for the current task
    def add(self, message: dict) -> None: ...
    def as_messages(self) -> list[dict]: ...

class LongTermMemory:              # persistent: reports, watchlist, prediction log
    def save(self, key: str, value: dict) -> None: ...
    def search(self, query: str) -> list[dict]: ...
```

```python
# core/base_agent.py
class BaseAgent(ABC):
    name: str
    description: str               # used by the Orchestrator to route tasks

    def __init__(self, llm: LLMClient, memory: LongTermMemory | None = None):
        self.llm = llm
        self.tools = ToolRegistry()
        self.register_tools()

    @abstractmethod
    def system_prompt(self) -> str: ...
    @abstractmethod
    def register_tools(self) -> None: ...

    def run(self, task: Task) -> AgentResult:
        """THE agent loop:
           1. send system prompt + messages + tools to Claude
           2. if stop_reason == "tool_use": run every tool_use block,
              return ALL results in ONE user message, then loop
           3. stop at end_turn (or a max-iterations cap), build AgentResult"""
```

Write this loop yourself — it's the heart of every agent. The SDK's Tool Runner
(`client.beta.messages.tool_runner` + `@beta_tool`) is an option later if you'd rather not maintain it.

### `agents/wall_street/`

```python
# agent.py
class WallStreetAgent(BaseAgent):
    name = "wall_street"
    description = "Equity research, market analysis, and price-move forecasting"

    def system_prompt(self) -> str: ...
    def register_tools(self) -> None: ...

    def research_company(self, ticker: str) -> CompanyReport: ...
    def recommend(self, tickers: list[str], risk_profile: str) -> list[StockRecommendation]: ...
    def market_outlook(self) -> str: ...
```

```python
# schemas.py (Pydantic)
class CompanyReport(BaseModel):
    ticker: str
    summary: str
    business_model: str
    financial_health: str
    valuation: str
    risks: list[str]
    catalysts: list[str]
    sources: list[str]

class PricePrediction(BaseModel):
    ticker: str
    horizon_days: int
    prob_up: float
    model_version: str
    model_backtest_accuracy: float  # so Claude knows how much to trust it

class StockRecommendation(BaseModel):
    ticker: str
    action: Literal["buy", "hold", "sell", "avoid"]
    confidence: Literal["low", "medium", "high"]
    thesis: str
    key_risks: list[str]
    prediction: PricePrediction | None
```

```python
# data/providers.py
class MarketDataProvider(ABC):
    def get_price_history(self, ticker, start, end, interval="1d") -> pd.DataFrame: ...
    def get_quote(self, ticker) -> dict: ...
    def get_financials(self, ticker) -> dict: ...

class YFinanceProvider(MarketDataProvider): ...    # free, prototyping
class AlphaVantageProvider(MarketDataProvider): ... # free tier, official API
class SECEdgarClient: ...                          # 10-K / 10-Q filings
class FREDClient: ...                              # macro data
```

Tools are thin `Tool` subclasses that call a provider or the ML code and return compact JSON.
For news/web research, Anthropic's server-side web search tool (`web_search_20260209`) is an
alternative to writing a scraper.

### `agents/wall_street/ml/`

```python
class FeatureBuilder:
    def build(self, prices: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
        """returns, volatility, RSI, MACD, volume z-scores, MA gaps, market/sector returns...
           RULE: every feature at row t uses only data available at t."""

class DatasetBuilder:
    def make_labels(self, prices, horizon: int = 5) -> pd.Series: ...  # forward return > 0
    def time_split(self, X, y, train_end, val_end) -> tuple: ...       # never shuffle

class BasePredictor(ABC):
    def fit(self, X, y) -> None: ...
    def predict_proba(self, X) -> np.ndarray: ...
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> "BasePredictor": ...

class NaiveBaselinePredictor(BasePredictor): ...   # must beat this
class LogisticPredictor(BasePredictor): ...
class GradientBoostPredictor(BasePredictor): ...   # LightGBM/XGBoost
class LSTMPredictor(BasePredictor): ...            # optional

class Trainer:
    def walk_forward(self, model, X, y, n_splits) -> dict: ...

class Backtester:
    def run(self, predictions, prices, cost_bps=5) -> BacktestReport: ...
    # accuracy, precision, Sharpe, max drawdown, vs. buy-and-hold

class ModelRegistry:
    def save(self, model, metrics: dict) -> str: ...
    def load_best(self, target: str) -> BasePredictor: ...
```

**Starter datasets:** yfinance daily history for S&P 500 constituents (10–20 years), FRED macro
series, Kaggle stock datasets (e.g. "Huge Stock Market Dataset", S&P 500 daily datasets).

## Roadmap

### Phase 0 — Setup
- [ ] `git init`, `.gitignore` (`.venv`, `.env`, `data/`, `models/`)
- [ ] `pyproject.toml`: `anthropic`, `pydantic`, `python-dotenv`, `pandas`, `yfinance`
- [ ] Rename to lowercase packages; create `core/` and `agents/wall_street/`
- **Milestone:** a script sends "hello" to Claude via the SDK

### Phase 1 — Core framework
- [ ] `LLMClient`, `Tool`, `ToolRegistry`, `BaseAgent.run()`
- [ ] Throwaway `EchoAgent` with one fake tool (e.g. `get_time`)
- **Milestone:** Claude calls your tool, gets the result, answers; tests pass

### Phase 2 — WallStreetAgent v1: research (no ML)
- [ ] Data providers + cache; market data, fundamentals, technicals, news, macro tools
- [ ] System prompt: analyst persona, cite data, state uncertainty, never invent numbers
- [ ] Structured `CompanyReport` output
- **Milestone:** `python main.py research NVDA` produces a sourced report from real tool data

### Phase 3 — Prediction model (training)
- [ ] Download/cache datasets; `FeatureBuilder`, `DatasetBuilder`
- [ ] Baselines first, then gradient boosting, with walk-forward validation
- [ ] Backtest with trading costs vs. buy-and-hold
- [ ] Prototype in `notebooks/`, then move working code into `ml/`
- **Milestone:** saved model with honest out-of-sample metrics (or proof it doesn't beat baseline)

### Phase 4 — Connect the model to the agent
- [ ] `PredictPriceMoveTool` loads the best model from `ModelRegistry`
- [ ] Claude gets the probability *and* the model's backtest accuracy
- [ ] `StockRecommendation` combines fundamentals, technicals, news, prediction
- **Milestone:** `recommend(["AAPL","MSFT","NVDA"])` returns ranked, reasoned recommendations

### Phase 5 — Memory & tracking
- [ ] `LongTermMemory`: watchlist, past reports, prediction log
- [ ] Paper-trading tracker that checks each prediction after its horizon
- **Milestone:** live accuracy view of the agent's past calls

### Phase 6 — Multi-agent
- [ ] `Orchestrator` + `AgentMessage` protocol; routing by agent `name`/`description`
- [ ] Future agents: NewsSentimentAgent, RiskManagerAgent
- **Milestone:** "Should I buy NVDA?" → WallStreetAgent consults RiskManagerAgent before answering

### Phase 7 — Quality
- [ ] Eval set of ~20 standard questions (cites sources, flags risks, no made-up numbers)
- [ ] Cost tracking per task; prompt caching for system prompt + tool definitions

## Pitfalls
- **Look-ahead bias:** future data leaking into features or normalisation.
- **Random train/test splits** on time series — always split by date.
- **Survivorship bias:** training only on companies that exist today.
- **Hallucinated numbers:** require every figure to come from a tool result.
