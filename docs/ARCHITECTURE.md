# MyAgents Architecture

What every class in the project does, why it's built that way, and how to use it as a
template for the next agent.

## The big picture: three layers

```
┌───────────────────────────────────────────────────────────────┐
│ main.py: the user interface (CLI)                             │
├───────────────────────────────────────────────────────────────┤
│ agents/wall_street/: WHAT this agent knows and can do         │
│   prompt · tools · data providers · output schemas            │
├───────────────────────────────────────────────────────────────┤
│ core/: HOW any agent works (reused by every future agent)     │
│   loop · LLM access · tool system · memory · data objects     │
└───────────────────────────────────────────────────────────────┘
```

**The main rule:** `core/` never imports from `agents/`. That one-way dependency is what
lets every future agent reuse all of `core/` unchanged.

## What happens when you run `python main.py research NVDA`

```
main.py → WallStreetAgent.research_company("NVDA")
            → builds a Task, calls BaseAgent.run(task, output_model=CompanyReport)
               loop:
                 LLMClient.create(system prompt, messages, tool definitions)  → Claude
                 Claude: "call get_quote(NVDA), get_fundamentals(NVDA), ..."  (stop_reason = tool_use)
                 ToolRegistry.execute(...) → each Tool.run() → YFinanceProvider → Yahoo
                 results go back into the conversation → loop again
                 Claude: final JSON report                                    (stop_reason = end_turn)
            → JSON validated into a CompanyReport → AgentResult → printed
```

---

## `core/`: the reusable framework

### `Settings` (config.py)
**Purpose:** one place for every setting: model, effort, max tokens, iteration cap, data
folder. Values come from `.env` or environment variables.
**Why:** without it, things like `"claude-opus-5-5"` end up hardcoded in many files. Now you
change the model in one place, or per run with `AGENT_MODEL=...`.
**Next agent:** reuse as is.

### Data objects (messages.py)
These are plain containers with no logic. They define what goes into and comes out of an
agent, so agents, the CLI and the future Orchestrator can all talk to each other without
knowing each other's internals.

| Class | Purpose |
|---|---|
| `Task` | The input: an objective, optional context (like risk profile), who asked, and an ID. Every agent accepts the same `Task`, so in the future one agent can hand work to another. |
| `ToolCall` | One audit-trail entry: which tool, what input, whether it failed, and a preview of the output. This is how you check *what the agent actually looked at*, which is essential for debugging and trust. |
| `AgentResult` | The output: final text, parsed structured output, the tool calls, token usage and cost, and the iteration count. It reports both what the agent said and how it got there. |
| `AgentMessage` | An envelope for agent-to-agent communication. A placeholder for the Orchestrator phase. |

### `LLMClient` (llm.py)
**Purpose:** the only class that talks to the Anthropic API.
**Why wrap the SDK instead of calling it directly?**
- **Testability:** tests replace it with `FakeLLM`, which has the same `create()` method, so
  they run with no API key and no cost.
- **One place for everything that applies to every call:** streaming, prompt caching,
  effort, refusal fallbacks and error translation.
- **Swappability:** changing models or providers touches one file.

### `UsageTracker` (llm.py)
**Purpose:** adds up tokens across every API call in one task and estimates the cost. That's
where the `~$0.xxx` line at the end of a run comes from.
**Why:** an agent makes many calls per task, and costs grow quietly. In production, cost per
task is a metric people track.

### `Tool` (tools.py)
**Purpose:** the template every tool follows: a `name`, a `description`, an `input_schema`,
and a `run()` method.
**Why it matters:** the model only sees three things about a tool: its name, description and
schema. It uses those to decide when and how to call it. So writing tool descriptions is
basically prompt engineering, and it's where most of an agent's quality comes from.
**Design choice:** `strict: True` makes the API guarantee Claude's input matches your schema,
so `run()` never receives malformed arguments.

### `ToolRegistry` (tools.py)
**Purpose:** holds an agent's tools, hands their definitions to the API, and **executes tool
calls safely**.
**The key design:** `execute()` never raises. If a tool fails, the error goes back to Claude
as a `tool_result` with `is_error: true`. Claude reads it ("No data for 'ZZZZQ'. Is the
ticker symbol correct?") and can fix its input or work around it, so the agent recovers
without crashing.
It also caps output size, since one huge tool result would be re-read and paid for on every
later loop iteration.

### `to_strict_json_schema()` (schema_utils.py)
**Purpose:** converts a Pydantic model into the strict JSON Schema format the API requires
for structured outputs.
**Why it exists:** Pydantic's default schema has `$ref` links and open objects, and the API
rejects both. This is the kind of glue code you only discover by hitting the error.

### `ConversationMemory` (memory.py)
**Purpose:** short-term memory, meaning the message list for one task.
**Key lesson:** the API is **stateless**. Every loop iteration resends the whole
conversation, so this list *is* the agent's memory. It's append-only and stores the full
response content (thinking blocks and tool calls included), because editing history breaks
tool-call matching and preserved thinking.

### `LongTermMemory` (memory.py)
**Purpose:** persists results across runs (`data/memory.json`). Research reports are saved
there.
**Why so simple:** it's a JSON file with substring search. The interface (`save` and
`search`) is what matters. Later you can swap in SQLite or a vector database without changing
any agent code.

### `BaseAgent` (base_agent.py): the most important class
**Purpose:** the agent loop. Every agent inherits it.
**It owns:** calling the model, running tools, sending results back, handling each
`stop_reason`, enforcing the iteration cap, validating structured output, and building the
`AgentResult`.
**Subclasses only define:** `system_prompt()` and `register_tools()`.
This is the **Template Method pattern**: the base class fixes the algorithm, and subclasses
fill in the parts that differ. Expect to explain this one in interviews.
**Two entry points:** `run(task)` creates a fresh `ConversationMemory` and throws it away
afterwards (one-shot commands). `run_turn(memory, task)` runs the same loop on a memory you
pass in, which is what chat mode uses.

### `ChatSession` (chat.py): multi-turn chat with any agent
**Purpose:** follow-up questions (`python main.py chat`). It lives in `core/`, so every
future agent gets chat mode for free.
**How it works:** a chat is just **one `ConversationMemory` kept alive across questions**.
Each time you type, it appends your message and calls `agent.run_turn(memory)`. The model
remembers nothing itself; we resend the whole conversation each turn.
**What it handles:**
- **Failed turns roll back.** A turn that dies halfway (error, refusal, Ctrl+C) can leave
  a `tool_use` with no `tool_result`, and the API would then reject every later request.
  `send()` records the memory length first and calls `memory.rollback()` on any failure,
  so the chat carries on as if the bad question was never asked.
- **Cost stays low through caching.** Turn N resends turns 1 to N−1, but because history is
  append-only, that prefix is identical each time and comes from the prompt cache at about
  1/10 the price. In the first smoke test, turn 1 cost $0.019 and the follow-up cost $0.005.
- **Context growth.** It tracks the conversation size, warns past 120k tokens, and `/reset`
  starts fresh. The production-grade next step is **compaction**: summarizing old turns
  into a short note.
- **The date goes in the first message only**, so later messages are sent exactly as typed.

REPL commands: `/help`, `/reset`, `/cost`, `/tools`, `/exit`.

### Error classes: one per layer

| Error | Raised by | What happens |
|---|---|---|
| `DataUnavailable` | the data provider | converted to a `ToolError` |
| `ToolError` | a tool | **sent back to Claude**, which can recover |
| `LLMError` | `LLMClient` | stops the run; it's a problem for the user (bad key, network) |
| `AgentError` | `BaseAgent` | stops the run; the agent itself failed (refusal, iteration cap) |

In chat mode, `LLMError` and `AgentError` only cancel the current question: `ChatSession`
rolls the conversation back and the chat continues.

The principle: **errors the model can fix go back to the model, and errors only you can fix
stop the program.** That's a reusable design decision for any agent.

### `Orchestrator` (orchestrator.py)
Not built yet (roadmap phase 6). It will route a `Task` to the right agent using each agent's
`name` and `description`, and deliver `AgentMessage`s between agents.

---

## `agents/wall_street/`: the domain layer

### `WallStreetAgent` (agent.py)
**Purpose:** gives the generic loop a role. It defines the prompt, the tool list, and four
jobs: `research_company`, `recommend`, `market_outlook` and `ask`.
**Notice:** it's about 100 lines and contains no loop logic. Each job just builds a `Task`
(and optionally picks an output schema), then calls `run()`. **The job methods are where you
encode expertise:** the objective text tells the agent what a good research report covers.

### `SYSTEM_PROMPT` (prompts.py)
**Purpose:** the agent's identity and rules.
**Decisions worth copying:**
- Every rule comes with a reason ("your training data is out of date, so a remembered figure
  is likely wrong"). Models follow rules better when they know why.
- No dates or volatile content, which keeps it in the prompt cache.
- It doesn't describe the tools, because their own descriptions already reach the model.

### `MarketDataProvider` (abstract) and `YFinanceProvider` (providers.py)
**Purpose:** the abstract class defines *what data exists*, and `YFinanceProvider` defines
*where it comes from*.
**Why the split:** tools depend on the abstract class, not on yfinance. That means:
- tests use `FakeProvider`, with no network,
- you can switch to a paid data source later without touching any tool.

This is **dependency inversion**, another interview favorite.

### `ttl_cache` (cache.py)
**Purpose:** remembers provider results for a few minutes. During one task Claude asks for
the same ticker's data several times (quote, then history, then technicals, which also needs
history), and this avoids re-downloading it.

### The 8 tools (tools/)

| Tool | Gives Claude | Design note |
|---|---|---|
| `GetQuoteTool` | price, day change, 52-week range, market cap | anchors the analysis in the current price |
| `GetPriceHistoryTool` | returns, volatility, drawdown, about 25 sampled prices | **summarizes** instead of dumping 250 rows, to save context |
| `GetMarketOverviewTool` | index ETFs, VIX, 10-year yield, gold, dollar | one failing symbol doesn't break the whole result |
| `GetFundamentalsTool` | sector, business summary, about 25 key ratios | renames Yahoo's field names to clear ones |
| `GetFinancialStatementsTool` | revenue, profit and cash-flow trends over years | picks key lines, not whole statements |
| `ComparePeersTool` | side-by-side ratios for 2–8 tickers | relative valuation ("expensive vs. what?") |
| `ComputeTechnicalsTool` | moving averages, RSI, MACD, trend, with readings | math in a pure function, reusable for ML features |
| `GetNewsTool` | 10 headlines with source and date | the description says headlines aren't facts |

Helpers: `clean_ticker()` validates input before any network call, and `provider_call()`
translates `DataUnavailable` into `ToolError`.
**General rule:** each tool is a **thin wrapper**: validate input, call a provider or pure
function, return a small dict. Keep logic out of tools so it's testable.

### Schemas (schemas.py)

| Class | Purpose |
|---|---|
| `CompanyReport` | the shape of a research report. Field descriptions double as instructions to the model. |
| `StockRecommendation` | one buy/hold/sell view. `Literal` types force valid values like `"buy"`, never `"strong buy!!"`. |
| `RecommendationSet` | several recommendations plus portfolio notes |
| `Metric` | `{name, value}` pairs, because structured outputs don't allow free-form dicts |

**Why structured output:** results become *data*. You can store them, compare them over
time, and test them, instead of parsing text.

---

## `tests/`

| Class | Purpose |
|---|---|
| `FakeLLM` | replays scripted Claude responses, to test loop logic for free and deterministically |
| `FakeProvider` | synthetic price data, to test tools with no network |

**Lesson:** you test agent *logic* with fakes and agent *quality* with evals (roadmap phase
7). They're different things.

---

## Checklist for the next agent

**Reuse unchanged:** everything in `core/`.

**Write new, in `agents/<new_agent>/`:**
1. **Jobs:** what tasks will it do? Those become methods like `research_company`.
2. **Output:** for each job, should the result be free text or a schema? Use a schema if
   you'll store, compare or act on it.
3. **Tools:** what information or actions does it need? Write one class per tool, as thin
   wrappers with clear descriptions and small outputs.
4. **Data sources:** where does the data come from? Put an abstract provider in front of it
   if you might swap the source or need to test without network.
5. **System prompt:** role, rules with reasons, and how to handle uncertainty.
6. **Risk level:** does any tool *change* something, like sending an email, placing a trade
   or deleting a file? If so, add a human approval step before it runs. WallStreetAgent only
   reads data, which is why it didn't need one.
7. **Model and effort:** heavy reasoning means a higher `effort`. Simple, high-volume work
   could use a cheaper model.

**Then ask yourself:** does it need to talk to WallStreetAgent? If so, that's when you build
the Orchestrator.

## Interview topics in this code
- The tool-use loop and its stop reasons
- Strict tool schemas
- Returning errors to the model instead of crashing
- Prompt caching (stable prefix, volatile data at the end)
- Structured outputs instead of parsing text
- Dependency inversion (`MarketDataProvider`)
- Template Method pattern (`BaseAgent`)
- Testing agents with a fake LLM vs. measuring quality with evals
