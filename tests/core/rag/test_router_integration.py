"""Integration tests for the RAG Router — requires live embedding provider.

Run with::

    pytest tests/core/rag/test_router_integration.py -v -s

Prerequisites:
    EMBED_BASE_URL, EMBED_MODEL (and EMBED_API_KEY if auth needed)
    LLM_BASE_URL, LLM_MODEL (and LLM_API_KEY if auth needed — for the LLMClient wrapper)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from three_kingdoms_ai_agent.core.config import LLMConfig, RAGSettings, Settings
from three_kingdoms_ai_agent.core.llm.client import LLMClient
from three_kingdoms_ai_agent.core.rag.router import RouteResult, Router

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router(
    memes_path: str | None = None,
    tmp_path: Path | None = None,
    threshold: float = 0.75,
) -> Router:
    """Build a Router from the live EMBED_* env vars.

    Uses ``data/memes.yaml`` by default, or a temp path for isolated tests.
    """
    config = LLMConfig.from_env()
    if not config.embed.base_url:
        pytest.skip("EMBED_BASE_URL (or LLM_BASE_URL) not set — skipping")
    if not config.embed.model:
        pytest.skip("EMBED_MODEL (or LLM_MODEL) not set — skipping")

    client = LLMClient(config)

    db_path = str(tmp_path / "memes.db") if tmp_path else ":memory:"

    settings = Settings(
        rag=RAGSettings(
            similarity_threshold=threshold,
            top_k=3,
            db_path=db_path,
            embed_batch_size=10,
        ),
    )

    memes = memes_path or str(Path("data/memes.yaml"))
    return Router.from_config(client, settings, memes_path=memes)


# ---------------------------------------------------------------------------
# Router integration
# ---------------------------------------------------------------------------


class TestRouterIntegration:
    """End-to-end tests for meme routing with real embeddings."""

    def test_from_config_populates_store(self, tmp_path):
        """After from_config(), the store should contain all memes."""
        router = _make_router(tmp_path=tmp_path)
        assert router._store.count() > 0, "Store should have memes after from_config"

    def test_exact_meme_phrase_hits_high_similarity(self, tmp_path):
        """An exact meme phrase from the corpus should match with high similarity."""
        router = _make_router(tmp_path=tmp_path)

        hit = router.route("当浮一大白")
        assert hit is not None, "Exact meme phrase must match"
        assert hit.agent_id == "recipe_agent"
        assert hit.sub_type == "喝什么"
        assert hit.similarity > 0.8, (
            f"Exact phrase should have high sim, got {hit.similarity:.4f}"
        )

    def test_short_common_phrase_hits(self, tmp_path):
        """A short, common meme phrase ('吃什么') should match."""
        router = _make_router(tmp_path=tmp_path)

        hit = router.route("吃什么")
        assert hit is not None, "Short meme '吃什么' must match"
        assert hit.agent_id == "recipe_agent"
        assert hit.sub_type == "吃什么"

    def test_unrelated_text_misses(self, tmp_path):
        """Completely unrelated text should not match any meme."""
        router = _make_router(tmp_path=tmp_path)

        hit = router.route("今天天气真好适合去爬山")
        assert hit is None, "Unrelated text should not match any meme"

    def test_route_result_fields_are_populated(self, tmp_path):
        """A successful route result should have all fields populated and valid."""
        router = _make_router(tmp_path=tmp_path)

        hit = router.route("天意")
        assert hit is not None
        assert isinstance(hit, RouteResult)
        assert isinstance(hit.agent_id, str) and hit.agent_id
        assert isinstance(hit.sub_type, str) and hit.sub_type
        assert isinstance(hit.meme_text, str) and hit.meme_text
        assert isinstance(hit.similarity, float)
        assert 0.0 <= hit.similarity <= 1.0

    def test_idempotent_from_config(self, tmp_path):
        """Calling from_config() twice on the same db should not double-populate."""
        router1 = _make_router(tmp_path=tmp_path)
        count1 = router1._store.count()

        router2 = _make_router(tmp_path=tmp_path)
        count2 = router2._store.count()

        assert count2 == count1, (
            f"Idempotent from_config: count changed from {count1} to {count2}"
        )
