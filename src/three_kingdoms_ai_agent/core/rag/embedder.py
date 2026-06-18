"""Embedder — thin wrapper around :class:`LLMClient` embedding methods.

Keeps the Router decoupled from the concrete LLM client so the embedding
backend can be swapped (e.g. local model, different provider) without
touching routing logic.
"""

from __future__ import annotations


class Embedder:
    """Thin wrapper around :class:`~core.llm.client.LLMClient` for embedding calls.

    Usage::

        embedder = Embedder(llm_client)
        vec = embedder.embed("当浮一大白")
        vecs = embedder.embed_batch(["台词1", "台词2"])
    """

    def __init__(self, client) -> None:
        """*client* must expose ``embed(text) -> list[float]`` and
        ``embed_batch(texts) -> list[list[float]]`` (the
        :class:`~core.llm.client.LLMClient` interface).
        """
        self._client = client

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Returns
        -------
        list[float]
            The embedding vector (dimension depends on the configured model).
        """
        return self._client.embed(text)

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
        """
        if not texts:
            return []
        return self._client.embed_batch(texts)
