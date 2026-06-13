"""LLM client — unified OpenAI-compatible interface for chat and embedding."""

from .action import Action, ActionType
from .client import ChatResult, LLMClient, LLMError
from .parser import parse_structured

__all__ = [
    "Action",
    "ActionType",
    "ChatResult",
    "LLMClient",
    "LLMError",
    "parse_structured",
]
