"""Tool base class + ToolRegistry.

A "tool" is just a Python function Claude can ask us to run. Claude never runs code
itself: it replies with a `tool_use` block ({name, input}); WE execute it and send
the output back as a `tool_result`. The model only ever sees:
  * name + description + input_schema  (to decide WHEN and HOW to call it)
  * whatever text we return             (to reason about the result)

So tool quality = description quality + output quality. Keep outputs small and
relevant: every byte you return is read (and paid for) on every later loop iteration.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, ClassVar
import json
import logging
import math

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 12_000  # hard cap so one tool can't flood the context window


class ToolError(Exception):
    """Raise from a tool for an expected failure (bad ticker, no data...).
    The message is sent back to Claude, which can then correct itself."""


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[dict[str, Any]]

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Do the work. Return anything JSON-serialisable (or a string)."""

    def to_api_dict(self) -> dict[str, Any]:
        # strict=True makes the API guarantee Claude's tool input matches the schema
        # exactly. That requires additionalProperties: false and every field required.
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # Anthropic-hosted tools (e.g. web search) run on Anthropic's servers; we only
        # declare them. Their results arrive inside the model's response.
        self._server_tools: list[dict[str, Any]] = []

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} registered twice")
        self._tools[tool.name] = tool

    def register_server_tool(self, definition: dict[str, Any]) -> None:
        self._server_tools.append(definition)

    def get(self, name: str) -> Tool:
        return self._tools[name]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def api_definitions(self) -> list[dict[str, Any]]:
        # Order is deterministic (registration order) - changing it would bust the
        # prompt cache, since tools are the very start of the cached prefix.
        return [tool.to_api_dict() for tool in self._tools.values()] + self._server_tools

    def execute(self, name: str, tool_input: dict[str, Any], tool_use_id: str) -> dict[str, Any]:
        """Run one tool and wrap the outcome as a `tool_result` content block.

        Never raises: errors go back to Claude with is_error=True so it can retry
        with different input or work around the problem.
        """
        tool = self._tools.get(name)
        if tool is None:
            return _result(tool_use_id, f"Unknown tool {name!r}. Available: {self.names}", True)
        try:
            output = tool.run(**tool_input)
            return _result(tool_use_id, to_json_text(output), False)
        except ToolError as e:
            return _result(tool_use_id, f"Error: {e}", True)
        except Exception as e:  # unexpected bug in our tool code
            log.exception("Tool %s crashed", name)
            return _result(tool_use_id, f"Tool crashed: {type(e).__name__}: {e}", True)


def _result(tool_use_id: str, content: str, is_error: bool) -> dict[str, Any]:
    if len(content) > MAX_RESULT_CHARS:
        content = content[:MAX_RESULT_CHARS] + "\n...[truncated]"
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def to_json_text(value: Any) -> str:
    """JSON-encode tool output, coping with pandas/numpy types, NaN and dates."""
    if isinstance(value, str):
        return value
    return json.dumps(_clean(value), ensure_ascii=False, separators=(",", ":"))


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()[:10]
    if hasattr(value, "item"):  # numpy scalar -> python scalar
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 4)
    return value
