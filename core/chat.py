"""ChatSession: multi-turn conversation with any agent.

The API is STATELESS: the model remembers nothing between requests. "Memory" in a chat
is just us resending the whole conversation every time. So a chat is simply:

    memory = ConversationMemory()          # created ONCE, kept for the whole session
    each time the user types something:
        memory.add_user(text)
        agent.run_turn(memory)             # the normal agent loop, on the same memory

That's the entire trick. Compare BaseAgent.run(), which creates a fresh memory per call
and throws it away - which is why one-shot commands can't answer "what about AMD?".

Three things a real chat has to handle, and how this class does it:

1. Failed turns. If a turn dies halfway (network error, Ctrl+C, iteration cap), the
   memory may end with a tool_use that never got a tool_result. The API rejects that
   conversation forever after. Fix: remember the length before the turn and roll back
   on any failure, so the session is exactly as it was before the bad turn.

2. Cost growth. Turn N resends turns 1..N-1, so input tokens grow every turn. Prompt
   caching (enabled in LLMClient) makes the repeated prefix ~10x cheaper, and it works
   BECAUSE history is append-only: the earlier messages are byte-for-byte identical
   each time. Editing old messages would break the cache.

3. Context limit. The conversation can't grow forever. We track its size and warn;
   /reset starts over. (The production-grade fix is "compaction": summarize old turns
   into a short note - a good exercise to build next.)
"""

from dataclasses import dataclass, field

from core.base_agent import BaseAgent
from core.memory import ConversationMemory
from core.messages import AgentResult, Task

# Past this size, each turn gets noticeably slower and pricier; suggest a fresh start.
CONTEXT_WARN_TOKENS = 120_000

USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
              "estimated_cost_usd")


@dataclass
class ChatSession:
    agent: BaseAgent
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    turns: int = 0
    totals: dict = field(default_factory=lambda: dict.fromkeys(USAGE_KEYS, 0))
    context_tokens: int = 0

    def send(self, text: str) -> AgentResult:
        """Send one user message; returns the agent's reply. On any failure the session
        is rolled back to how it was before this message, then the error is re-raised."""
        text = text.strip()
        if not text:
            raise ValueError("Empty message.")
        task = Task(objective=text, requested_by="chat")

        # Only the FIRST message gets the date header (build_user_message). Later turns
        # are sent as typed - repeating the header would just waste tokens.
        content = self.agent.build_user_message(task) if self.turns == 0 else text

        checkpoint = len(self.memory)
        self.memory.add_user(content)
        try:
            result = self.agent.run_turn(self.memory, task)
        except BaseException:          # BaseException so Ctrl+C mid-turn is undone too
            self.memory.rollback(checkpoint)
            raise

        self.turns += 1
        self.context_tokens = result.context_tokens
        for key in USAGE_KEYS:
            self.totals[key] += result.usage.get(key, 0)
        return result

    def reset(self) -> None:
        """Forget the conversation (the agent and its tools stay loaded)."""
        self.memory.clear()
        self.turns = 0
        self.context_tokens = 0

    @property
    def context_is_large(self) -> bool:
        return self.context_tokens > CONTEXT_WARN_TOKENS
