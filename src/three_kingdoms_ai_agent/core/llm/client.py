"""LLM client — synchronous, OpenAI-compatible, with structured-output parsing.

Provides :class:`LLMClient` for chat and embedding calls, and
:class:`ChatResult` to distinguish structured (action) responses from
free-form conversational text.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from .action import Action
from .parser import parse_structured
from ..config import LLMConfig

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Raised when an LLM API call fails (network, auth, rate-limit, etc.)."""

    def __init__(self, message: str, original: Exception | None = None) -> None:
        super().__init__(message)
        self.original = original


# ---------------------------------------------------------------------------
# ChatResult
# ---------------------------------------------------------------------------


@dataclass
class ChatResult:
    """The result of an :meth:`LLMClient.chat` call.

    Attributes
    ----------
    content : str
        The raw text returned by the LLM (always present).
    action : Action | None
        A validated structured action if the LLM returned a well-formed
        JSON control signal; ``None`` for free-form text.
    """

    content: str
    action: Action | None = None

    @property
    def is_structured(self) -> bool:
        """``True`` when the response contains a valid action."""
        return self.action is not None


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Synchronous, OpenAI-compatible LLM client.

    Credentials and model names come from :class:`~core.config.LLMConfig`;
    operational settings (timeout, max_retries) are passed explicitly.

    Usage::

        config = LLMConfig.from_env()
        client = LLMClient(config, timeout=60, max_retries=3)
        result = client.chat([{"role": "user", "content": "你好"}])
        print(result.content)
    """

    def __init__(
        self,
        config: LLMConfig,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """Initialize the OpenAI SDK client.

        Parameters
        ----------
        config : LLMConfig
            Provider configuration from environment variables.
        timeout : int
            HTTP request timeout in seconds (from ``config/llm.yaml``).
        max_retries : int
            Max retries on transient failures (from ``config/llm.yaml``).
        """
        self._config = config

        # Ollama and other local providers don't need auth; the OpenAI SDK
        # requires *some* api_key value though, so we pass a placeholder.
        api_key = config.api_key if config.auth_enabled else "ollama"

        self._client = OpenAI(
            base_url=config.base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    # -- chat -----------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> ChatResult:
        """Send a chat completion request and return a structured result.

        The caller is responsible for assembling the full ``messages`` list,
        including any system prompt.

        Parameters
        ----------
        messages : list[dict]
            List of message dicts with ``"role"`` and ``"content"`` keys.
        temperature : float
            Sampling temperature (default 0.3 — conservative, suitable for
            the main orchestrator; sub-agents may pass higher values).

        Returns
        -------
        ChatResult
            Always contains the raw ``content`` string.  If the LLM returned
            valid action JSON, the ``action`` field will be populated.

        Raises
        ------
        LLMError
            On any API or network failure.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
            )
        except Exception as exc:
            raise LLMError(
                f"LLM chat request failed: {exc}", original=exc
            ) from exc

        content = response.choices[0].message.content or ""

        # Attempt structured parsing
        parsed = parse_structured(content)
        action = Action.from_dict(parsed) if parsed else None

        return ChatResult(content=content, action=action)

    # -- embedding ------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        list[float]
            The embedding vector.

        Raises
        ------
        LLMError
            On any API or network failure.
        """
        try:
            response = self._client.embeddings.create(
                model=self._config.embed_model,
                input=text,
            )
        except Exception as exc:
            raise LLMError(
                f"LLM embedding request failed: {exc}", original=exc
            ) from exc

        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        list[list[float]]
            Embedding vectors in the same order as *texts*.

        Raises
        ------
        LLMError
            On any API or network failure.
        """
        try:
            response = self._client.embeddings.create(
                model=self._config.embed_model,
                input=texts,
            )
        except Exception as exc:
            raise LLMError(
                f"LLM batch embedding request failed: {exc}", original=exc
            ) from exc

        # Sort by index to preserve input order
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [item.embedding for item in sorted_data]
