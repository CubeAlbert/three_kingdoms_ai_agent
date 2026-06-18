"""Window memory — sliding window of recent conversation turns.

Uses :class:`collections.deque` with a fixed max length so memory usage stays
bounded regardless of conversation length.
"""

from __future__ import annotations

from collections import deque

from .base import MemoryManager


class WindowMemory(MemoryManager):
    """Sliding-window conversation memory backed by :class:`~collections.deque`.

    Stores each turn as a ``{"role": str, "content": str}`` dict.  When the
    window is full the oldest turn is silently dropped.

    Parameters
    ----------
    window_size : int
        Max number of turns to retain (default 10, i.e. 5 Q&A pairs).

    Usage::

        mem = WindowMemory(window_size=4)
        mem.add("user", "A")
        mem.add("assistant", "B")
        mem.add("user", "C")
        mem.add("assistant", "D")
        mem.add("user", "E")   # oldest ("user", "A") evicted
        assert len(mem.get_context()) == 4
    """

    def __init__(self, window_size: int = 10) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self._window_size = window_size
        self._turns: deque[dict] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # MemoryManager interface
    # ------------------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        """Append one turn.  Oldest turn is evicted if the window is full."""
        self._turns.append({"role": role, "content": content})

    def get_context(self, limit: int | None = None) -> list[dict]:
        """Return turns in chronological order.

        Parameters
        ----------
        limit : int | None
            Max most-recent turns to return.  ``None`` returns all.
        """
        turns = list(self._turns)
        if limit is not None and limit < len(turns):
            return turns[-limit:]
        return turns

    def store_long_term(self, key: str, value: object) -> None:
        """No-op — window memory is transient."""

    def recall_long_term(self, key: str) -> object | None:
        """No-op — always returns ``None``."""
        return None

    def clear(self) -> None:
        """Drop all stored turns."""
        self._turns.clear()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of turns currently stored."""
        return len(self._turns)

    def __repr__(self) -> str:
        return (
            f"WindowMemory(window_size={self._window_size}, "
            f"turns={len(self._turns)})"
        )
