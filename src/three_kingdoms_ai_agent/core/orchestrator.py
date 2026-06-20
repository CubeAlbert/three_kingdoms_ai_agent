"""Orchestrator — main agent loop with deterministic routing and template rendering.

The orchestrator ties together the Channel, LLM, RAG Router, Memory, and
sub-agents into a single interactive loop.  It follows the "deterministic-first"
principle: RAG routing is a cosine-similarity threshold check (no LLM), and hit
integration uses Python string templates (no LLM).  Only the miss (chat) path
and the sub-agent's content generation involve an LLM call.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..agents.base import AgentContext, AgentResult, BaseAgent
from .channel.base import AgentResponse, Channel
from .llm.client import ChatResult, LLMClient
from .memory.base import MemoryManager
from .rag.router import RouteResult, Router

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persona & Chat Rules — used in the RAG-miss (chat) path
# ---------------------------------------------------------------------------

PERSONA = (
    '你是新三国世界的军师，足智多谋，善于言辞。'
    '你既能与来客谈天说地，也能调度人手处理具体事务。'
    '说话半文半白，带三国演义风格，时而引用典故，时而豪迈痛快。'
    '自称「某」或「本军师」。'
)

CHAT_RULES = (
    "当来客的话不涉及具体事务时，你以三国军师的身份与他闲聊。"
    "你可以讨论三国典故、兵法谋略、天下大势。"
    "如果来客的话让你联想到某个梗或台词，你可以调侃几句。"
    "保持角色，不要跳出三国背景。回复应简洁有力，不可啰嗦，控制在三句话以内。"
)

# ---------------------------------------------------------------------------
# Exit keywords — checked on every user input
# ---------------------------------------------------------------------------

_EXIT_KEYWORDS = frozenset({"exit", "quit", "q", "退出", "告辞", "拜别"})


def _is_exit(text: str) -> bool:
    """Return ``True`` when *text* is a shutdown signal."""
    return text.strip().lower() in _EXIT_KEYWORDS


# ---------------------------------------------------------------------------
# Template rendering — deterministic hit integration (no LLM)
# ---------------------------------------------------------------------------

# Per-agent templates.  Each value is a callable ``(AgentResult) -> str``.
# The template receives the full AgentResult so it can branch on sub_type
# or data fields.  Agents that are not listed here fall through to
# ``_fallback_template``.

def _recipe_template(result: AgentResult) -> str:
    """Render RecipeAgent structured output in the strategist's voice."""
    d = result.data or {}
    name = d.get("name", "???")
    reason = d.get("reason", "天机不可泄露")
    description = d.get("description", "")
    suggestion = d.get("suggestion", "")

    if result.sub_type == "喝什么":
        intro = f"军师举杯笑道：「说到杯中物，某倒是想起一物——**{name}**，{reason}」"
    else:
        intro = f"军师抚须沉吟道：「既是腹中饥馑，某有一计——**{name}**，{reason}」"

    parts = [intro]
    if description:
        parts.append(f"\n{description}")
    if suggestion:
        parts.append(f"\n「{suggestion}」")

    return "".join(parts)


def _fallback_template(result: AgentResult) -> str:
    """Generic template used when no per-agent template is registered."""
    if not result.data:
        return f"军师曰：「{result.raw_content}」"
    # Dump the structured data as a readable list
    lines = ["军师捋须道：「据探子来报——"]
    for key, value in result.data.items():
        lines.append(f"  · {key}：{value}")
    lines.append("」")
    return "\n".join(lines)


# agent_id → template callable
_DEFAULT_TEMPLATES: dict[str, Callable[[AgentResult], str]] = {
    "recipe_agent": _recipe_template,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Main agent loop.

    Ties together Channel, LLM, RAG Router, Memory, and sub-agents.
    Follows the deterministic-first principle: RAG routing is a cosine
    threshold check, and hit integration uses Python string templates.

    Usage::

        orchestrator = Orchestrator(
            channel=CliChannel(),
            llm=llm_client,
            router=router,
            memory=WindowMemory(window_size=10),
            agents={"recipe_agent": RecipeAgent()},
        )
        orchestrator.run()
    """

    def __init__(
        self,
        channel: Channel,
        llm: LLMClient,
        router: Router,
        memory: MemoryManager,
        agents: dict[str, BaseAgent],
        *,
        templates: dict[str, Callable[[AgentResult], str]] | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Parameters
        ----------
        channel : Channel
            I/O transport (e.g. :class:`CliChannel`).
        llm : LLMClient
            Shared LLM client for chat and embedding.
        router : Router
            Pre-built meme router (usually from :meth:`Router.from_config`).
        memory : MemoryManager
            Conversation memory (window + long-term stub).
        agents : dict[str, BaseAgent]
            Registry mapping ``agent_id`` → sub-agent instance.  Keys must
            match the ``agent_id`` values in ``data/memes.yaml``.
        templates : dict[str, Callable[[AgentResult], str]] | None
            Optional per-agent template overrides.  Merged with the built-in
            defaults (caller wins on conflict).
        """
        self._channel = channel
        self._llm = llm
        self._router = router
        self._memory = memory
        self._agents = agents

        # Merge caller templates over defaults
        merged = dict(_DEFAULT_TEMPLATES)
        if templates:
            merged.update(templates)
        self._templates = merged

    # -- main loop -------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive conversation loop.

        Blocks until the user sends an exit signal (e.g. "quit", "退出")
        or EOF (Ctrl+D / Ctrl+Z).
        """
        self._channel.send(AgentResponse(
            "军师已至，有何见教？（输入「退出」或「告辞」以辞别军师）"
        ))

        while True:
            # --- receive -------------------------------------------------------
            try:
                msg = self._channel.receive()
            except EOFError:
                self._channel.send(AgentResponse("军师告退，后会有期！"))
                break

            if not msg.text:
                continue  # skip empty input

            # --- exit check ----------------------------------------------------
            if _is_exit(msg.text):
                self._channel.send(AgentResponse("军师告退，后会有期！"))
                break

            # --- memory --------------------------------------------------------
            self._memory.add("user", msg.text)

            # --- route (deterministic) -----------------------------------------
            route_result = self._router.route(msg.text)

            if route_result is not None:
                # HIT — sub-agent → structured result → template render
                logger.info(
                    "RAG HIT | agent=%s sub_type=%s similarity=%.3f meme=%r",
                    route_result.agent_id,
                    route_result.sub_type,
                    route_result.similarity,
                    route_result.meme_text,
                )
                response_text = self._handle_hit(route_result, msg.text)
            else:
                # MISS — persona + chat rules + LLM (no json_mode)
                logger.info("RAG MISS — using chat mode")
                response_text = self._handle_miss(msg.text)

            # --- memory + respond ----------------------------------------------
            self._memory.add("assistant", response_text)
            self._channel.send(AgentResponse(response_text))

    # -- hit path --------------------------------------------------------------

    def _handle_hit(self, route: RouteResult, user_message: str) -> str:
        """Route to a sub-agent, then template-render its structured result.

        This path makes exactly **one** LLM call — inside the sub-agent's
        :meth:`BaseAgent.handle`.  Result integration is deterministic.
        """
        agent = self._agents.get(route.agent_id)
        if agent is None:
            logger.warning(
                "No agent registered for agent_id=%r — falling back to chat.",
                route.agent_id,
            )
            return self._handle_miss(user_message)

        ctx = AgentContext(
            user_message=user_message,
            sub_type=route.sub_type,
            matched_meme=route.meme_text,
            history=self._memory.get_context(),
            llm=self._llm,
            memory=self._memory,
        )

        try:
            logger.info("Switching to sub-agent: %s (sub_type=%s)", agent.name, route.sub_type)
            result = agent.handle(ctx)
        except Exception:
            logger.exception(
                "Sub-agent %s raised an exception — falling back to chat.",
                agent.name,
            )
            return self._handle_miss(user_message)

        return self._render_hit(result)

    # -- miss path -------------------------------------------------------------

    def _handle_miss(self, user_message: str) -> str:
        """Chat mode — persona + chat rules + history + user message → LLM.

        This path makes exactly **one** LLM call without ``json_mode``
        (free-form conversational text).
        """
        messages: list[dict] = [
            {"role": "system", "content": PERSONA + "\n\n" + CHAT_RULES},
        ]
        messages.extend(self._memory.get_context())
        messages.append({"role": "user", "content": user_message})

        try:
            result: ChatResult = self._llm.chat(
                messages, temperature=0.8, json_mode=False
            )
            return result.content
        except Exception:
            logger.exception("LLM chat failed — returning canned response.")
            return "军师一时语塞……「此事容某再思量思量。」"

    # -- template rendering ----------------------------------------------------

    def _render_hit(self, result: AgentResult) -> str:
        """Render a sub-agent's structured result with a deterministic template.

        Supports per-agent templates (via the ``templates`` dict passed at
        init).  Falls back to :func:`_fallback_template` when no template is
        registered or when *result.data* is ``None``.
        """
        template = self._templates.get(result.agent_id, _fallback_template)
        try:
            return template(result)
        except Exception:
            logger.exception(
                "Template rendering failed for agent_id=%r — using fallback.",
                result.agent_id,
            )
            return _fallback_template(result)
