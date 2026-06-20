"""Sub-agent base class with three-layer prompt assembly and structured output.

Defines :class:`BaseAgent` — the contract every sub-agent fulfills — along with
:class:`AgentContext` (input) and :class:`AgentResult` (output).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.llm.client import LLMClient
from ..core.memory.base import MemoryManager

# ---------------------------------------------------------------------------
# JSON mode safety net
# ---------------------------------------------------------------------------
# DeepSeek and OpenAI both require the word "json" somewhere in the prompt
# when ``response_format={'type': 'json_object'}`` is set — otherwise the
# API returns a 400 error.  Each sub-agent's ``system_prompt`` should include
# its own JSON schema example; this suffix exists only as a fallback and is
# appended when the assembled system prompt does not contain "json".

_JSON_FALLBACK_SUFFIX = "\n\nYou MUST respond with a valid JSON object."


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------


@dataclass
class AgentContext:
    """All the information a sub-agent needs to handle one user request.

    Populated by the orchestrator after a successful RAG route match.
    """

    user_message: str
    """The raw text the user typed."""

    sub_type: str
    """Sub-type identifier from RAG metadata (deterministic)."""

    matched_meme: str
    """The specific meme phrase that triggered the route."""

    history: list[dict]
    """Recent conversation turns as ``{"role": ..., "content": ...}`` dicts,
    sourced from :meth:`MemoryManager.get_context`."""

    llm: LLMClient
    """The shared LLM client (chat + embedding)."""

    memory: MemoryManager
    """The shared memory manager (window + long-term stub)."""


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Structured output from a sub-agent's :meth:`BaseAgent.handle` call.

    The orchestrator uses this for deterministic template rendering — no
    second LLM call is needed for integration.
    """

    agent_id: str
    """The :attr:`BaseAgent.name` of the agent that produced this result."""

    sub_type: str
    """The sub-type that was used for prompt selection."""

    data: dict | None
    """The parsed structured data (schema varies per sub-agent), or ``None``
    if parsing failed."""

    raw_content: str
    """The original LLM response text, always preserved for debugging."""

    @property
    def is_ok(self) -> bool:
        """``True`` when structured data was successfully parsed."""
        return self.data is not None


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Sub-agent contract.

    Each sub-agent holds:

    * **system_prompt** — the agent's role / persona / language style
      (invariant).  **Must contain the word "json" and an example JSON
      output structure** so that ``json_mode=True`` works correctly with
      DeepSeek / OpenAI.
    * **sub_type_prompts** — a deterministic ``{sub_type: prompt}`` mapping;
      the orchestrator supplies ``sub_type`` via :class:`AgentContext`, and
      the agent selects the matching processing prompt.

    The default :meth:`handle` implementation performs three-layer prompt
    assembly (System + Sub-Type + User) → one LLM call with
    ``json_mode=True`` → :meth:`parse_result`.  Subclasses typically only
    need to override :meth:`parse_result` and define their prompts.

    Usage::

        class RecipeAgent(BaseAgent):
            name = "recipe_agent"
            description = "Recommend dishes and drinks, Three-Kingdoms style"
            system_prompt = "你是新三国的随军厨子...\\n输出 JSON 格式：..."
            sub_type_prompts = {
                "吃什么": "用户问今天吃什么...",
                "喝什么": "用户问喝什么...",
            }

            def parse_result(self, raw_content, ctx):
                data = json.loads(raw_content)
                return AgentResult(
                    agent_id=self.name,
                    sub_type=ctx.sub_type,
                    data=data,
                    raw_content=raw_content,
                )
    """

    # -- sub-class overrides ---------------------------------------------------

    name: str = ""
    """Unique agent identifier (matches ``agent_id`` in ``data/memes.yaml``)."""

    description: str = ""
    """Human-readable summary of what this agent does."""

    system_prompt: str = ""
    """The agent's role persona, language style, and output format.

    **Must include the word "json" and an example JSON output structure**
    so that ``json_mode=True`` works correctly with DeepSeek / OpenAI.
    """

    sub_type_prompts: dict[str, str] = {}
    """Deterministic mapping from ``sub_type`` → processing prompt.

    Keys must match the ``sub_type`` values in ``data/memes.yaml``.
    """

    temperature: float = 0.7
    """Sampling temperature for this agent's LLM calls (default 0.7).

    Sub-agents that need more determinism (e.g. structured extraction) can
    lower this; creative agents can raise it.
    """

    # -- public API ------------------------------------------------------------

    def handle(self, ctx: AgentContext) -> AgentResult:
        """Execute the three-layer prompt → LLM → parse pipeline.

        Parameters
        ----------
        ctx : AgentContext
            The request context from the orchestrator.

        Returns
        -------
        AgentResult
            Structured output; ``data`` is ``None`` when parsing fails.
        """
        messages = self._build_messages(ctx)

        # JSON mode safety net: DeepSeek / OpenAI require "json" in the prompt
        # when json_mode=True.  Only apply this check in the default handle()
        # path — conversational sub-agents that override handle() and use
        # json_mode=False do NOT need this.
        has_json = any(
            "json" in (msg.get("content", "") or "").lower() for msg in messages
        )
        if not has_json:
            messages[0]["content"] += _JSON_FALLBACK_SUFFIX

        result = ctx.llm.chat(
            messages, temperature=self.temperature, json_mode=True
        )
        return self.parse_result(result.content, ctx)

    @abstractmethod
    def parse_result(self, raw_content: str, ctx: AgentContext) -> AgentResult:
        """Parse the LLM's raw JSON output into a structured :class:`AgentResult`.

        Subclasses must implement this — each agent defines its own output
        schema.

        Parameters
        ----------
        raw_content : str
            The raw text returned by the LLM (should be valid JSON when
            ``json_mode=True`` is used).
        ctx : AgentContext
            The original request context (provides ``sub_type`` etc.).

        Returns
        -------
        AgentResult
            The parsed result.  Set ``data=None`` when parsing fails so
            the orchestrator can fall back gracefully.
        """
        ...

    # -- internal helpers ------------------------------------------------------

    def _build_messages(self, ctx: AgentContext) -> list[dict]:
        """Assemble the full message list for the LLM call.

        Three layers (deterministic assembly):

        1. **System** — ``system_prompt`` + selected ``sub_type_prompt``
        2. **History** — recent conversation turns from ``ctx.history``
        3. **User** — the current ``ctx.user_message``
        """
        sub_prompt = self.sub_type_prompts.get(ctx.sub_type, "")
        system = self._assemble_system_prompt(sub_prompt)

        messages: list[dict] = [{"role": "system", "content": system}]
        messages.extend(ctx.history)
        messages.append({"role": "user", "content": ctx.user_message})
        return messages

    def _assemble_system_prompt(self, sub_prompt: str) -> str:
        """Combine ``system_prompt`` + *sub_prompt* (pure concatenation).

        No JSON fallback is appended here — that responsibility moved to
        :meth:`handle` which runs only in the ``json_mode=True`` path.
        Conversational sub-agents that override :meth:`handle` with
        ``json_mode=False`` are unaffected.
        """
        parts = [self.system_prompt]
        if sub_prompt:
            parts.append(sub_prompt)
        return "\n\n".join(parts)
