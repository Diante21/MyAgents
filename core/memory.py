"""Memory for agents.

ConversationMemory: short-term. The `messages` list for one task. The API is
    stateless, so the whole list is sent on every loop iteration.
LongTermMemory:     persistent across runs. A simple JSON file for now (past reports,
    watchlist). Later you could swap in SQLite or a vector store behind the same API.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


class ConversationMemory:
    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []

    def add_user(self, content: Any) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: Any) -> None:
        # IMPORTANT: store the FULL content list from the response (thinking blocks,
        # tool_use blocks, text) - not just the text. The API needs the tool_use
        # blocks to match tool_results, and thinking blocks must be passed back
        # unchanged. Never edit earlier messages: history is append-only.
        self._messages.append({"role": "assistant", "content": content})

    def as_messages(self) -> list[dict[str, Any]]:
        return self._messages

    def __len__(self) -> int:
        return len(self._messages)

    def rollback(self, length: int) -> None:
        """Drop everything after the first `length` messages. The ONE exception to
        append-only: undoing a turn that failed halfway. A failed turn can leave a
        tool_use with no tool_result, and the API rejects any conversation containing
        one - so without this, one error would break the rest of the chat."""
        del self._messages[length:]

    def clear(self) -> None:
        self._messages.clear()


class LongTermMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, list[dict[str, Any]]] = (
            json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        )

    def save(self, collection: str, record: dict[str, Any]) -> None:
        record = {"saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        self._data.setdefault(collection, []).append(record)
        self.path.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")

    def search(self, collection: str, contains: str = "") -> list[dict[str, Any]]:
        """Naive substring search. Good enough until you need semantic search."""
        needle = contains.lower()
        return [r for r in self._data.get(collection, []) if needle in json.dumps(r).lower()]
