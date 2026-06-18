"""Channel abstraction — decouples agent core from I/O transport."""

from .base import AgentResponse, Channel, Message

__all__ = [
    "AgentResponse",
    "Channel",
    "Message",
]
