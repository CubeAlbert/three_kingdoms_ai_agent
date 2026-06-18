"""Long-term memory — no-op stub with a complete interface.

Sub-agents can call :meth:`store_long_term` and :meth:`recall_long_term` from
day one; the calls are silently accepted.  In Milestone 3 this module will be
replaced with a real implementation (vector DB + summarisation) without any
code changes in the agents or orchestrator.
"""

from __future__ import annotations

from .base import MemoryManager


class LongTermMemory(MemoryManager):
    """No-op long-term memory that accepts all calls and returns ``None``.

    Implements the full :class:`MemoryManager` interface so sub-agents and the
    orchestrator can depend on the ABC without knowing which concrete backend
    is wired in.

    ``add`` / ``get_context`` / ``clear`` are also no-ops — long-term storage
    does not double as the conversation window.  Use :class:`WindowMemory` for
    the sliding window.
    """

    def add(self, role: str, content: str) -> None:
        """No-op — long-term memory doesn't track conversation turns."""

    def get_context(self, limit: int | None = None) -> list[dict]:
        """No-op — always returns an empty list."""
        return []

    def store_long_term(self, key: str, value: object) -> None:
        """No-op — payload accepted but not persisted.

        Sub-agents can call this freely; the call is a silent no-op in MVP.
        """

    def recall_long_term(self, key: str) -> object | None:
        """No-op — always returns ``None``.

        Sub-agents that check the return value before using it will naturally
        skip the missing data path.
        """
        return None

    def clear(self) -> None:
        """No-op — nothing to clear."""

    def __repr__(self) -> str:
        return "LongTermMemory()"
