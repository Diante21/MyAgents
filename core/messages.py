# Task, AgentResult, AgentMessage dataclasses
from dataclasses import dataclass


@dataclass
class Task:
    id: str
    objective: str               # "Research NVDA and give a buy/hold/sell view"
    context: str                 # extra inputs (portfolio, risk tolerance, etc.)
    requested_by: str            # "user" or another agent's name


@dataclass
class AgentResult:
    task_id: str
    agent: str
    output: str | BaseModel          # text or a structured schema object
    tool_calls : list[dict]          # audit trail: what the agent actually looked at
    usage : dict | None              # tokens / cost



@dataclass
class AgentMessage:                 # agent-to-agent communication (later phase)
    sender: str
    recipient: str
    task: Task



