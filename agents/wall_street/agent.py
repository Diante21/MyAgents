"""WallStreetAgent: equity research on top of the generic BaseAgent loop.

Notice how little code is here. All the agent-loop machinery lives in core/; this
class only answers three questions:
  1. Who am I?            -> system_prompt()
  2. What can I use?      -> register_tools()
  3. What jobs do I do?   -> research_company / recommend / market_outlook / ask
"""

from core.base_agent import BaseAgent
from core.llm import LLMClient
from core.memory import LongTermMemory
from core.messages import AgentResult, Task
from core.config import settings
from agents.wall_street.data.providers import MarketDataProvider, YFinanceProvider
from agents.wall_street.ml.registry import ModelRegistry
from agents.wall_street.portfolio.providers import PortfolioProvider, provider_for
from agents.wall_street.tools.portfolio import AnalyzePortfolioRiskTool, GetPortfolioTool
from agents.wall_street.prompts import SYSTEM_PROMPT
from core.base_agent import AgentError
from agents.wall_street.schemas import CompanyReport, PortfolioReview, RecommendationSet
from agents.wall_street.tools.fundamentals import (
    ComparePeersTool, GetFinancialStatementsTool, GetFundamentalsTool,
)
from agents.wall_street.tools.market_data import (
    GetMarketOverviewTool, GetPriceHistoryTool, GetQuoteTool,
)
from agents.wall_street.tools.news import GetNewsTool
from agents.wall_street.tools.predict import PredictPriceMoveTool
from agents.wall_street.tools.technicals import ComputeTechnicalsTool


class WallStreetAgent(BaseAgent):
    name = "wall_street"
    description = "Equity research, company analysis, market overview, and stock recommendations"

    def __init__(
        self,
        llm: LLMClient,
        provider: MarketDataProvider | None = None,
        memory: LongTermMemory | None = None,
        web_search: bool = False,
        model_registry: ModelRegistry | None = None,
        portfolio: PortfolioProvider | None = None,
        **kwargs,
    ):
        # Set these BEFORE super().__init__, because it calls register_tools().
        self.provider = provider or YFinanceProvider()
        self.web_search = web_search
        self.model_registry = model_registry or ModelRegistry(settings.models_dir)
        # Portfolio is optional: no file -> no portfolio tools, everything else still works.
        if portfolio is None and settings.portfolio_file.exists():
            portfolio = provider_for(settings.portfolio_file)
        self.portfolio = portfolio
        super().__init__(llm=llm, memory=memory, **kwargs)

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def register_tools(self) -> None:
        for tool_cls in (
            GetQuoteTool,
            GetPriceHistoryTool,
            GetMarketOverviewTool,
            GetFundamentalsTool,
            GetFinancialStatementsTool,
            ComparePeersTool,
            ComputeTechnicalsTool,
            GetNewsTool,
        ):
            self.tools.register(tool_cls(self.provider))

        # Only offer the prediction tool once a model has been trained. Offering a tool
        # that can only fail wastes a model call (and confuses the agent).
        if self.model_registry.latest_version() is not None:
            self.tools.register(PredictPriceMoveTool(self.provider, self.model_registry))

        # Read-only portfolio tools. Note what's NOT here: no tool can place an order.
        if self.portfolio is not None:
            self.tools.register(GetPortfolioTool(self.portfolio, self.provider))
            self.tools.register(AnalyzePortfolioRiskTool(self.portfolio, self.provider))

        if self.web_search:
            # Anthropic-hosted web search: runs on Anthropic's servers, billed per search.
            # Useful for things our data tools lack (earnings call commentary, regulation,
            # product launches). max_uses caps cost per request.
            self.tools.register_server_tool(
                {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}
            )

    # ---- the agent's jobs ---------------------------------------------------------
    def research_company(self, ticker: str) -> AgentResult:
        task = Task(
            objective=(
                f"Write a research report on {ticker.upper()}. Cover the business, financial "
                "health, valuation versus 2-4 close peers, the technical picture, and recent "
                "news, then the bull and bear cases."
            ),
        )
        result = self.run(task, output_model=CompanyReport)
        self._remember("reports", ticker, result)
        return result

    def recommend(self, tickers: list[str], risk_profile: str = "moderate") -> AgentResult:
        task = Task(
            objective=(
                f"Give a buy/hold/sell/avoid view on each of: {', '.join(t.upper() for t in tickers)}. "
                "Research each one before deciding, and rank them best idea first."
            ),
            context={"risk_profile": risk_profile, "time_horizon": "medium (3-12 months) unless a ticker clearly warrants otherwise"},
        )
        result = self.run(task, output_model=RecommendationSet)
        self._remember("recommendations", ",".join(tickers), result)
        return result

    def market_outlook(self) -> AgentResult:
        return self.run(Task(objective=(
            "Give a concise market outlook: where major indexes, volatility, rates, gold and the "
            "dollar stand, what that suggests about current risk appetite, and what to watch next."
        )))

    def review_portfolio(self, risk_profile: str = "moderate") -> AgentResult:
        if self.portfolio is None:
            raise AgentError(
                f"No portfolio file found at {settings.portfolio_file}. Export a Robinhood "
                "activity report (or fill in docs/portfolio.example.csv) and save it there, "
                "or set PORTFOLIO_FILE in .env.")
        task = Task(
            objective=(
                "Review my current portfolio. Look at the holdings, how concentrated and risky "
                "the portfolio is versus the S&P 500, and research the largest positions enough "
                "to judge them. Then give an overall assessment with strengths, concerns, and "
                "ideas for me to consider. I make all trading decisions myself."
            ),
            context={"risk_profile": risk_profile},
        )
        result = self.run(task, output_model=PortfolioReview)
        self._remember("portfolio_reviews", "portfolio", result)
        return result

    def ask(self, question: str) -> AgentResult:
        """Free-form question - the agent decides which tools it needs."""
        return self.run(Task(objective=question))

    def _remember(self, collection: str, key: str, result: AgentResult) -> None:
        if self.memory and result.output is not None:
            self.memory.save(collection, {"key": key, "result": result.output.model_dump()})
