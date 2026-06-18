"""Channel abstraction — decouples agent core from I/O transport.

Agent core never touches stdin / stdout / HTTP directly; it only talks to a
:class:`Channel`.  This makes it cheap to swap CLI, WebSocket, or WeChat
transports later without touching any orchestration logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Message — what the user sends in
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A normalized user message received through a :class:`Channel`.

    Attributes
    ----------
    content : str
        The raw text the user typed (or the semantic equivalent from a
        non-text channel, e.g. a voice transcription).
    metadata : dict
        Optional channel-level metadata (e.g. ``{"source": "wechat",
        "user_id": "abc"}``).  Not part of the conversation content itself but
        may be forwarded to the orchestrator or memory layer.
    """

    content: str
    metadata: dict = field(default_factory=dict, repr=False)

    @property
    def text(self) -> str:
        """Convenience alias for :attr:`content`."""
        return self.content


# ---------------------------------------------------------------------------
# AgentResponse — what the agent sends back
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """A structured response from an agent, to be delivered through a :class:`Channel`.

    Attributes
    ----------
    content : str
        The human-readable text to display to the user.
    metadata : dict
        Optional structured payload (e.g. ``{"action": ..., "target": ...}``)
        that the channel or frontend may use for rich rendering.  Sub-agents
        attach their structured results here so the orchestrator can
        render them deterministically.
    """

    content: str
    metadata: dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Channel ABC
# ---------------------------------------------------------------------------


class Channel(ABC):
    """Abstract base for all I/O transports.

    Concrete channels handle the mechanics of getting text in and out; the
    orchestrator only sees :meth:`receive` → :class:`Message` and
    :meth:`send` ← :class:`AgentResponse`.

    MVP is synchronous (matching the rest of Phase 1); async support is
    deferred to Phase 3.
    """

    @abstractmethod
    def receive(self) -> Message:
        """Block until the user sends a message, then return it.

        Returns
        -------
        Message
            The normalized user input.
        """

    @abstractmethod
    def send(self, response: AgentResponse) -> None:
        """Deliver an agent response to the user.

        Parameters
        ----------
        response : AgentResponse
            The text (and optional metadata) to display.
        """
