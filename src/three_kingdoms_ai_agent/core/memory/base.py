"""Memory management abstract interface.

Defines :class:`MemoryManager` — the contract that all memory implementations
must fulfill.  The orchestrator and sub-agents depend only on this ABC so
concrete backends can be swapped without touching orchestration logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryManager(ABC):
    """Abstract interface for conversation memory management.

    MVP design (Phase 1):
      - :meth:`add` / :meth:`get_context` handle short-term window memory.
      - :meth:`store_long_term` / :meth:`recall_long_term` are no-op stubs
        with a complete interface, so sub-agents can call them now and get
        real behaviour later without code changes.

    All methods are **synchronous** — matching the rest of Phase 1.  Async
    support is deferred to Phase 3 (see ``docs/plan.md``).

    Usage::

        memory = WindowMemory(window_size=10)
        memory.add("user", "今天吃什么？")
        memory.add("assistant", "军师建议来碗面。")
        ctx = memory.get_context()
        # ctx → [{"role": "user", "content": "今天吃什么？"},
        #         {"role": "assistant", "content": "军师建议来碗面。"}]
    """

    @abstractmethod
    def add(self, role: str, content: str) -> None:
        """Store one conversation turn.

        Parameters
        ----------
        role : str
            The message role — typically ``"user"``, ``"assistant"``, or
            ``"system"`` (the same values the OpenAI chat API expects).
        content : str
            The message text.
        """

    @abstractmethod
    def get_context(self, limit: int | None = None) -> list[dict]:
        """Return recent conversation turns as chat-compatible dicts.

        Each entry is ``{"role": str, "content": str}`` — suitable for
        passing directly to :meth:`LLMClient.chat` as part of the
        ``messages`` list.

        Parameters
        ----------
        limit : int | None
            Max number of most-recent turns to return.  ``None`` (or
            exceeding the stored count) returns all available turns.

        Returns
        -------
        list[dict]
            Turns in chronological order (oldest first).
        """

    @abstractmethod
    def store_long_term(self, key: str, value: object) -> None:
        """Persist a long-term memory entry.

        **MVP**: no-op stub.  The interface is here so sub-agents can call
        it from day one; the real implementation (vector DB + summarisation)
        lands in Milestone 3.

        Parameters
        ----------
        key : str
            A stable lookup key (e.g. ``"user:preferences:cuisine"``).
        value : object
            Arbitrary JSON-serialisable payload.
        """

    @abstractmethod
    def recall_long_term(self, key: str) -> object | None:
        """Recall a long-term memory entry by key.

        **MVP**: always returns ``None``.

        Parameters
        ----------
        key : str
            The lookup key previously passed to :meth:`store_long_term`.

        Returns
        -------
        object | None
            The stored value, or ``None`` if the key is not found.
        """

    @abstractmethod
    def clear(self) -> None:
        """Drop all in-memory state (useful for testing and reset)."""
