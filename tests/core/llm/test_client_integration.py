"""Integration tests for LLMClient — require a live LLM provider.

Run with::

    pytest tests/core/llm/test_client_integration.py -v -s

Prerequisites:

    Chat:
        LLM_BASE_URL, LLM_MODEL, LLM_AUTH_ENABLED (and LLM_API_KEY if auth needed)

    Embedding:
        EMBED_BASE_URL, EMBED_MODEL, EMBED_AUTH_ENABLED (and EMBED_API_KEY if auth needed)
        — each falls back to its LLM_* counterpart when unset.
"""

from __future__ import annotations

import pytest

from three_kingdoms_ai_agent.core.config import LLMConfig
from three_kingdoms_ai_agent.core.llm.client import LLMClient
from three_kingdoms_ai_agent.core.llm.action import ActionType


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> LLMClient:
    """Create an LLMClient from live environment variables.

    Raises pytest.skip if the required env vars are not set.
    """
    config = LLMConfig.from_env()
    if not config.base_url:
        pytest.skip("LLM_BASE_URL not set — skipping integration test")
    return LLMClient(config)


def _require_embed_config(config: LLMConfig) -> None:
    """Skip the test if the embedding provider is not configured."""
    if not config.embed.base_url:
        pytest.skip("EMBED_BASE_URL (or LLM_BASE_URL) not set — skipping embedding test")
    if not config.embed.model:
        pytest.skip("EMBED_MODEL (or LLM_MODEL) not set — skipping embedding test")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class TestChatIntegration:
    def test_plain_chat_returns_content(self):
        """Free-form chat returns non-empty content without a structured action."""
        client = _make_client()
        result = client.chat(
            [{"role": "user", "content": "回复一个'你好'"}],
            temperature=0.1,
        )
        assert len(result.content) > 0
        assert result.action is None

    def test_json_mode_produces_structured_action(self):
        """With json_mode=True, the LLM returns a parseable action JSON."""
        client = _make_client()
        result = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是一个指令解析器。你的回答必须是一个纯JSON对象，不要有任何其他文字。\n"
                        '当你需要结束时，输出：{"action": "exit"}'
                    ),
                },
                {"role": "user", "content": "结束"},
            ],
            temperature=0.1,
            json_mode=True,
        )
        assert result.is_structured, (
            f"Expected structured action but got: {result.content}"
        )
        assert result.action.type == ActionType.EXIT

    def test_json_mode_respected_by_api(self):
        """json_mode=True must send response_format to the API (no 400 error)."""
        client = _make_client()
        # If the API doesn't support response_format or the prompt lacks "json",
        # the API would return 400 — this test verifies the path works end-to-end.
        result = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "输出纯JSON对象，格式：{\"action\": \"switch\", \"target\": \"<name>\"}"
                    ),
                },
                {"role": "user", "content": '切换到 recipe agent，输出 JSON'},
            ],
            temperature=0.1,
            json_mode=True,
        )
        # Must at least get a response (no crash)
        assert len(result.content) > 0
        # With json_mode + JSON instruction, should be structured
        assert result.is_structured, (
            f"Expected structured action but got: {result.content}"
        )
        assert result.action.type == ActionType.SWITCH
        assert result.action.target is not None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class TestEmbeddingIntegration:
    """Integration tests for LLMClient embedding — requires EMBED_* env vars."""

    def test_embed_returns_valid_vector(self):
        """embed() should return a non-empty list of floats."""
        client = _make_client()
        _require_embed_config(client._config)

        vec = client.embed("你好，测试一下")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_embed_batch_preserves_order_and_count(self):
        """embed_batch() should return one vector per input, in order."""
        client = _make_client()
        _require_embed_config(client._config)

        texts = ["当浮一大白", "天意", "关羽之歌"]
        vecs = client.embed_batch(texts)

        assert len(vecs) == 3
        for i, vec in enumerate(vecs):
            assert isinstance(vec, list), f"vec[{i}] is not a list"
            assert len(vec) > 0, f"vec[{i}] is empty"
            assert all(isinstance(v, float) for v in vec), f"vec[{i}] has non-float"

    def test_embed_batch_vectors_are_same_dimension(self):
        """All vectors returned by embed_batch() should have the same dimension."""
        client = _make_client()
        _require_embed_config(client._config)

        texts = ["a", "b", "c"]
        vecs = client.embed_batch(texts)

        dims = {len(v) for v in vecs}
        assert len(dims) == 1, f"Expected uniform dimension, got {dims}"

    def test_similar_texts_closer_than_dissimilar(self):
        """Semantically similar texts should have higher cosine similarity
        than clearly unrelated ones."""
        client = _make_client()
        _require_embed_config(client._config)

        # Texts that are clearly about the same topic
        topic_a = "我想喝酒"       # I want to drink
        topic_b = "来一杯好酒"     # Pour a good drink
        unrelated = "计算机编程语言"  # Programming languages — clearly different

        vecs = client.embed_batch([topic_a, topic_b, unrelated])
        v1, v2, v3 = vecs

        # Dot product on (approximately) L2-normalized embedding vectors
        def cos_sim(a, b):
            return sum(x * y for x, y in zip(a, b))

        sim_same_topic = cos_sim(v1, v2)
        sim_cross_a = cos_sim(v1, v3)
        sim_cross_b = cos_sim(v2, v3)

        assert sim_same_topic > sim_cross_a, (
            f"Same-topic sim ({sim_same_topic:.4f}) "
            f"> cross-topic sim A ({sim_cross_a:.4f})"
        )
        assert sim_same_topic > sim_cross_b, (
            f"Same-topic sim ({sim_same_topic:.4f}) "
            f"> cross-topic sim B ({sim_cross_b:.4f})"
        )
