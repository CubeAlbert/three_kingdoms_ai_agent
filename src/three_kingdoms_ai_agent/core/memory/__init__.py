"""Memory management — window memory and long-term memory stub.

Public API
----------
- :class:`MemoryManager` — abstract interface (``base.py``)
- :class:`WindowMemory` — sliding-window conversation memory (``window.py``)
- :class:`LongTermMemory` — no-op stub with complete interface (``long_term.py``)
"""

from __future__ import annotations

from .base import MemoryManager
from .long_term import LongTermMemory
from .window import WindowMemory

__all__ = ["MemoryManager", "WindowMemory", "LongTermMemory"]
