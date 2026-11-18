"""BaseAgent: the agent loop. Every agent in MyAgents inherits this.

The whole idea of an "agent" fits in one loop:

    messages = [user task]
    loop:
        response = model(system, messages, tools)
        append response to messages
        if the model asked for tools:   run them, append ALL results, loop again
        if the model is done:           return its answer

The model decides which tools to call and when to stop. Our code decides what tools
exist, enforces limits, and keeps an audit trail.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any
import logging

from pydantic import BaseModel, ValidationError

from core.config import settings
from core.llm import LLMClient
from core.memory import ConversationMemory, LongTermMemory
from core.messages import AgentResult, Task, ToolCall
from core.schema_utils import to_strict_json_schema
from core.tools import ToolRegistry

log = logging.getLogger(__name__)


class AgentError(RuntimeError):
    pass


class BaseAgent(ABC):
    name: str = "base"
    description: str = ""          # used later by the Orchestrator to route tasks

    def __init__(
        self,
        llm: LLMClient,
        memory: LongTermMemory | None = None,
        max_iterations: int = settings.max_iterations,
    ):
        self.llm = llm
        self.memory = memory
        self.max_iterations = max_iterations
        self.tools = ToolRegistry()
        self.register_tools()

    # ---- what each concrete agent must define -------------------------------------
    @abstractmethod
    def system_prompt(self) -> str:
        """Who the agent is and the rules it follows. Keep it STABLE across calls
        (no timestamps!) so it stays in the prompt cache."""

    @abstractmethod
    def register_tools(self) -> None:
        """Call self.tools.register(...) for every tool this agent can use."""

    # ---- the loop -----------------------------------------------------------------
    def run(self, task: Task, output_model: type[BaseModel] | None = None) -> AgentResult:
        """One-shot: a fresh conversation for this task, thrown away afterwards."""
        memory = ConversationMemory()
        memory.add_user(self.build_user_message(task))
        return self.run_turn(memory, task, output_model)

    def run_turn(self, memory: ConversationMemory, task: Task,
                 output_model: type[BaseModel] | None = None) -> AgentResult:
        """Run the loop on an EXISTING conversation whose last message is the user's.

        run() passes a brand-new memory; core/chat.py's ChatSession passes the same
        memory every turn - that one difference is what makes a multi-turn chat. The
        loop appends the model's replies and tool results to `memory` as it goes, so
        when this returns, `memory` holds the whole exchange, ready for the next turn.
        """
        schema = to_strict_json_schema(output_model) if output_model else None
        tools = self.tools.api_definitions()
        usage = self.llm.new_usage_tracker()
        audit: list[ToolCall] = []
        response = None

        for iteration in range(1, self.max_iterations + 1):
            response = self.llm.create(
                system=self.system_prompt(),
                messages=memory.as_messages(),
                tools=tools,
                output_schema=schema,
            )
            usage.add(response.usage)
            memory.add_assistant(response.content)
            stop = response.stop_reason
            log.debug("iteration %d stop_reason=%s", iteration, stop)

            if stop == "tool_use":
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    log.info("  -> %s(%s)", block.name, _short(block.input))
                    result = self.tools.execute(block.name, block.input, block.id)
                    results.append(result)
                    audit.append(ToolCall(
                        name=block.name,
                        input=dict(block.input),
                        is_error=result.get("is_error", False),
                        output_preview=str(result["content"])[:300],
                    ))
                # ALL results go back in ONE user message. Splitting them across
                # several messages teaches the model to stop making parallel calls.
                memory.add_user(results)
                continue

            if stop == "pause_turn":
                # A server-side tool (e.g. web search) paused a long turn. Sending the
                # conversation back as-is lets the server continue where it stopped.
                continue

            if stop == "end_turn":
                return self._finish(task, response, output_model, audit, usage, iteration)

            if stop == "max_tokens":
                raise AgentError("Response hit max_tokens. Raise AGENT_MAX_TOKENS.")
            if stop == "refusal":
                details = getattr(response, "stop_details", None)
                raise AgentError(f"The model declined this request: {details}")
            raise AgentError(f"Unexpected stop_reason: {stop}")

        raise AgentError(
            f"Gave up after {self.max_iterations} iterations without a final answer. "
            "The task may be too broad, or a tool keeps failing."
        )

    # ---- helpers subclasses may override -------------------------------------------
    def build_user_message(self, task: Task) -> str:
        """Turn a Task into the first user message. Volatile info (today's date) goes
        HERE rather than in the system prompt, so the system prompt stays cacheable."""
        lines = [f"Today's date: {date.today().isoformat()}", "", task.objective]
        if task.context:
            lines += ["", "Context:"] + [f"- {k}: {v}" for k, v in task.context.items()]
        return "\n".join(lines)

    def _finish(self, task, response, output_model, audit, usage, iterations) -> AgentResult:
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        parsed = None
        if output_model is not None:
            try:
                parsed = output_model.model_validate_json(text)
            except ValidationError as e:
                raise AgentError(f"Model output did not match {output_model.__name__}: {e}") from e
        u = response.usage
        return AgentResult(
            task_id=task.id,
            agent=self.name,
            text=text,
            output=parsed,
            tool_calls=audit,
            usage=usage.summary(),
            iterations=iterations,
            stop_reason=response.stop_reason,
            # Size of the conversation the model just read (all input, cached or not).
            # In a chat this grows every turn - it's how ChatSession knows when to warn.
            context_tokens=sum(getattr(u, k, 0) or 0 for k in
                               ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")),
        )


def _short(value: Any, limit: int = 80) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in value.items()) if isinstance(value, dict) else repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
