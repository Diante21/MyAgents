"""MyAgents command line.

Examples (from the project root, with the venv active):
    python main.py research NVDA
    python main.py recommend AAPL MSFT NVDA --risk moderate
    python main.py outlook
    python main.py ask "How has the semiconductor sector done versus the S&P 500 this year?"
    python main.py research TSLA --web          # also let the agent search the web
    python main.py holdings                     # print your holdings (no AI, free)
    python main.py portfolio --risk moderate    # agent reviews your holdings (read-only)
    python main.py chat                         # interactive: ask follow-up questions
"""

import argparse
import logging
import sys

from core.config import settings
from core.llm import LLMClient, LLMError
from core.base_agent import AgentError
from core.chat import ChatSession
from core.memory import LongTermMemory
from core.messages import AgentResult
from agents.wall_street import WallStreetAgent
from agents.wall_street.data.providers import YFinanceProvider
from agents.wall_street.portfolio.providers import PortfolioError, provider_for
from agents.wall_street.tools.portfolio import GetPortfolioTool


def print_result(result: AgentResult) -> None:
    print()
    if result.output is not None:
        print_structured(result.output.model_dump())
    else:
        print(result.text)

    u = result.usage
    print("\n" + "-" * 70)
    print(f"{len(result.tool_calls)} tool calls in {result.iterations} model calls | "
          f"{u['input_tokens'] + u['cache_read_tokens'] + u['cache_write_tokens']:,} input tokens "
          f"({u['cache_read_tokens']:,} from cache) | {u['output_tokens']:,} output tokens | "
          f"~${u['estimated_cost_usd']:.3f}")
    errors = [c for c in result.tool_calls if c.is_error]
    if errors:
        print(f"{len(errors)} tool call(s) returned errors: " + ", ".join(c.name for c in errors))


def print_structured(data, indent: int = 0) -> None:
    pad = "  " * indent
    for key, value in data.items():
        label = key.replace("_", " ").title()
        if isinstance(value, list) and value and isinstance(value[0], dict):
            print(f"{pad}{label}:")
            for item in value:
                if set(item) == {"name", "value"}:
                    print(f"{pad}  - {item['name']}: {item['value']}")
                else:
                    print_structured(item, indent + 1)
                    print()
        elif isinstance(value, list):
            print(f"{pad}{label}:")
            for item in value:
                print(f"{pad}  - {item}")
        else:
            print(f"{pad}{label}: {value}")


def show_holdings() -> int:
    """Read the portfolio file and print it with live prices. Uses the same tool the agent
    uses, but calls it directly - handy for checking your file before spending tokens."""
    try:
        portfolio = provider_for(settings.portfolio_file)
    except PortfolioError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    data = GetPortfolioTool(portfolio, YFinanceProvider()).run()

    def money(x):
        return "-" if x is None else f"${x:,.2f}"

    print(f"\nHoldings from {data['source']}" + (f" (activity through {data['data_as_of']})" if data["data_as_of"] else ""))
    print(f"{'Ticker':<8}{'Shares':>12}{'Avg cost':>12}{'Price':>12}{'Value':>14}{'Gain':>14}{'Gain %':>9}{'Weight':>8}  Sector")
    for p in data["positions"]:
        pct = "-" if p["unrealized_gain_pct"] is None else f"{p['unrealized_gain_pct']:+.1f}%"
        weight = "-" if p["weight_pct"] is None else f"{p['weight_pct']:.1f}%"
        print(f"{p['ticker']:<8}{p['shares']:>12,.4g}{money(p['avg_cost']):>12}{money(p['price']):>12}"
              f"{money(p['market_value']):>14}{money(p['unrealized_gain']):>14}{pct:>9}{weight:>8}  {p['sector']}")
    print(f"\nInvested {money(data['invested_value'])} | Cash {money(data['cash'])} | "
          f"Total {money(data['total_value'])} | Unrealized gain {money(data['total_unrealized_gain'])}")
    c = data["concentration"]
    if c:
        print(f"Effective number of holdings: {c['effective_number_of_holdings']} "
              f"(of {c['number_of_positions']}) | largest: {c['largest_position']} "
              f"{c['largest_position_pct_of_invested']}% of invested")
    for w in data["data_warnings"]:
        print(f"Note: {w}")
    return 0


CHAT_HELP = """Commands:
  /reset   forget the conversation and start fresh (cheaper once a chat gets long)
  /cost    tokens and estimated cost so far
  /tools   what the agent can look up
  /exit    quit (or Ctrl+C / Ctrl+D at the prompt)
Ctrl+C while the agent is working cancels that question only."""


def chat_loop(agent: WallStreetAgent) -> int:
    """REPL: Read a line, Evaluate it (one agent turn), Print the reply, Loop."""
    session = ChatSession(agent)
    print(f"\nWallStreetAgent chat ({settings.model}). Ask anything; follow-ups remember "
          f"the conversation.\nPortfolio: {'loaded' if agent.portfolio else 'none'} | "
          f"ML model: {'yes' if 'predict_price_move' in agent.tools.names else 'none'} | "
          f"/help for commands")

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):      # Ctrl+D / Ctrl+C at the prompt = quit
            print()
            break
        if not text:
            continue

        command = text.lower()
        if command in ("/exit", "/quit", "exit", "quit"):
            break
        if command == "/help":
            print(CHAT_HELP)
            continue
        if command == "/reset":
            session.reset()
            print("Conversation cleared.")
            continue
        if command == "/tools":
            print(", ".join(agent.tools.names))
            continue
        if command == "/cost":
            t = session.totals
            print(f"{session.turns} turns | {t['input_tokens'] + t['cache_read_tokens'] + t['cache_write_tokens']:,} "
                  f"input tokens ({t['cache_read_tokens']:,} from cache) | {t['output_tokens']:,} output | "
                  f"~${t['estimated_cost_usd']:.3f}")
            continue

        try:
            result = session.send(text)
        except KeyboardInterrupt:
            print("\n(cancelled - that question was removed from the conversation)")
            continue
        except (LLMError, AgentError) as e:
            # The session rolled itself back, so the chat can simply carry on.
            print(f"\nError: {e}\n(that question was removed from the conversation; you can keep chatting)",
                  file=sys.stderr)
            continue

        print(f"\nAgent: {result.text}")
        errors = sum(c.is_error for c in result.tool_calls)
        print(f"\n  [{len(result.tool_calls)} tool calls{f', {errors} failed' if errors else ''} | "
              f"~${result.usage['estimated_cost_usd']:.3f} this turn, "
              f"~${session.totals['estimated_cost_usd']:.3f} total | "
              f"conversation {session.context_tokens:,} tokens]")
        if session.context_is_large:
            print("  Tip: this conversation is getting long, so every turn resends a lot. "
                  "/reset starts fresh and cheaper.")

    print(f"Session total: {session.turns} turns, ~${session.totals['estimated_cost_usd']:.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a MyAgents agent.")
    parser.add_argument("--web", action="store_true", help="enable Anthropic web search (extra cost)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("research", help="research report on one company")
    p.add_argument("ticker")
    p = sub.add_parser("recommend", help="buy/hold/sell views on several tickers")
    p.add_argument("tickers", nargs="+")
    p.add_argument("--risk", default="moderate", choices=["conservative", "moderate", "aggressive"])
    sub.add_parser("outlook", help="overall market outlook")
    p = sub.add_parser("ask", help="free-form question")
    p.add_argument("question")
    p = sub.add_parser("portfolio", help="agent reviews your holdings (read-only)")
    p.add_argument("--risk", default="moderate", choices=["conservative", "moderate", "aggressive"])
    sub.add_parser("holdings", help="print your holdings with live prices (no AI, no cost)")
    sub.add_parser("chat", help="interactive conversation with follow-up questions")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    # Quiet noisy libraries; keep our own tool-call log lines.
    for noisy in ("httpx", "httpx2", "anthropic", "yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "holdings":
        return show_holdings()

    try:
        agent = WallStreetAgent(
            llm=LLMClient(),
            memory=LongTermMemory(settings.data_dir / "memory.json"),
            web_search=args.web,
        )
        if args.command == "chat":
            return chat_loop(agent)
        print(f"WallStreetAgent ({settings.model}, effort={settings.effort}) working...")
        if args.command == "research":
            result = agent.research_company(args.ticker)
        elif args.command == "recommend":
            result = agent.recommend(args.tickers, risk_profile=args.risk)
        elif args.command == "outlook":
            result = agent.market_outlook()
        elif args.command == "portfolio":
            result = agent.review_portfolio(risk_profile=args.risk)
        else:
            result = agent.ask(args.question)
    except (LLMError, AgentError, PortfolioError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
