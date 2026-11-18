"""LLMClient: the ONLY place in the project that imports `anthropic`.

Why wrap the SDK at all?
  * Agents stay testable: tests pass a fake client with the same `create()` method.
  * One place for cross-cutting concerns: model choice, effort, caching, refusal
    fallbacks, token/cost accounting, error translation.
  * Swapping models (or providers) later touches one file.
"""

from dataclasses import dataclass, field
from typing import Any
import logging

import anthropic

from core.config import settings

log = logging.getLogger(__name__)

# USD per 1M tokens: (input, output). Cache writes cost 1.25x input, cache reads 0.05x.
# Used only for the cost estimate we show the user — check the pricing page for truth.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class LLMError(RuntimeError):
    """Raised for failures the agent can't recover from (bad key, bad request...)."""


@dataclass
class UsageTracker:
    """Adds up tokens across every call an agent makes for one task."""
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    by_call: list[dict[str, int]] = field(default_factory=list)

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        entry = {
            "input": getattr(usage, "input_tokens", 0) or 0,
            "output": getattr(usage, "output_tokens", 0) or 0,
            "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        }
        self.calls += 1
        self.input_tokens += entry["input"]
        self.output_tokens += entry["output"]
        self.cache_read_tokens += entry["cache_read"]
        self.cache_write_tokens += entry["cache_write"]
        self.by_call.append(entry)

    @property
    def estimated_cost_usd(self) -> float:
        price_in, price_out = PRICES.get(self.model, PRICES["claude-opus-5-5"])
        return (
            self.input_tokens * price_in
            + self.cache_write_tokens * price_in * 1.25
            + self.cache_read_tokens * price_in * 0.05
            + self.output_tokens * price_out
        ) / 1_000_000

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
        }


class LLMClient:
    def __init__(
        self,
        model: str = settings.model,
        effort: str = settings.effort,
        max_tokens: int = settings.max_tokens,
        use_fallbacks: bool = True,
    ):
        if not settings.has_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        # The SDK reads ANTHROPIC_API_KEY itself and retries 429/5xx/connection
        # errors with exponential backoff (max_retries, default 2).
        self.client = anthropic.Anthropic(max_retries=3)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        # If a safety classifier declines a request (stop_reason "refusal"), the API can
        # transparently re-run it on another model. "default" lets the API pick the
        # fallback by refusal category. It needs the beta endpoint + a beta flag.
        self.use_fallbacks = use_fallbacks

    def create(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> Any:
        """One round-trip to Claude. Returns the SDK Message object."""
        output_config: dict[str, Any] = {"effort": self.effort}
        if output_schema is not None:
            # Constrains the FINAL text answer to JSON matching the schema.
            # Tool calls along the way are unaffected.
            output_config["format"] = {"type": "json_schema", "schema": output_schema}

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            "output_config": output_config,
            # Automatic prompt caching: the API caches the longest stable prefix
            # (tools -> system -> earlier messages). In an agent loop every call
            # resends the whole history, so this makes iterations 2..N much cheaper.
            "cache_control": {"type": "ephemeral"},
        }
        if tools:
            params["tools"] = tools
        # Thinking: claude-opus-5-5 always uses adaptive thinking, so we don't send a
        # `thinking` param at all. `effort` above is the knob.

        try:
            # Streaming avoids HTTP timeouts on long responses (big max_tokens).
            # get_final_message() assembles the stream into a normal Message.
            if self.use_fallbacks:
                with self.client.beta.messages.stream(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **params,
                ) as stream:
                    return stream.get_final_message()
            with self.client.messages.stream(**params) as stream:
                return stream.get_final_message()

        # Most specific first. The SDK already retried the retryable ones.
        except anthropic.AuthenticationError as e:
            raise LLMError("Invalid ANTHROPIC_API_KEY.") from e
        except anthropic.BadRequestError as e:
            raise LLMError(f"Bad request: {e.message}") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Rate limited even after retries. Try again in a minute.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the Anthropic API. Check your connection.") from e

    def new_usage_tracker(self) -> UsageTracker:
        return UsageTracker(model=self.model)
