"""Plain data objects that flow in and out of agents.

Keeping these separate from the agent code means the Orchestrator (later) can pass
Tasks between agents without knowing anything about how each agent works inside.
"""

from dataclasses import dataclass, field
from typing import Any
import uuid

from pydantic import BaseModel


@dataclass
class Task:
    objective: str                                   # "Research NVDA and give a buy/hold/sell view"
    context: dict[str, Any] = field(default_factory=dict)  # extra inputs (risk profile, portfolio...)
    requested_by: str = "user"                       # "user" or another agent's name
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class ToolCall:
    """One entry in the audit trail: what the agent looked at and what came back."""
    name: str
    input: dict[str, Any]
    is_error: bool
    output_preview: str                              # first part of the result, for logs/debugging


@dataclass
class AgentResult:
    task_id: str
    agent: str
    text: str                                        # the model's final text answer
    output: BaseModel | None = None                  # parsed structured output, if a schema was requested
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)  # tokens + estimated cost
    iterations: int = 0                              # how many model calls the loop made
    stop_reason: str | None = None
    context_tokens: int = 0                          # tokens the model read on its last call


@dataclass
class AgentMessage:
    """Agent-to-agent communication (used by the Orchestrator in a later phase)."""
    sender: str
    recipient: str
    task: Task
