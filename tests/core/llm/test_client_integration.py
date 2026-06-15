"""Integration tests for LLMClient — require a live LLM provider.

Run with::

    pytest tests/core/llm/test_client_integration.py -v -s

Prerequisites:
    LLM_BASE_URL, LLM_MODEL, LLM_AUTH_ENABLED (and LLM_API_KEY if auth needed)
    must be set in the environment.
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
