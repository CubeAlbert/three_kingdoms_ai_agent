"""Tests for core.llm.client — LLMClient, ChatResult, LLMError.

Integration tests (marked with ``@pytest.mark.integration``) require
valid environment variables and a live LLM provider.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from three_kingdoms_ai_agent.core.config import (
    ENV_LLM_API_KEY,
    ENV_LLM_AUTH_ENABLED,
    ENV_LLM_BASE_URL,
    ENV_LLM_EMBED_MODEL,
    ENV_LLM_MODEL,
    LLMConfig,
)
from three_kingdoms_ai_agent.core.llm.client import ChatResult, LLMClient, LLMError
from three_kingdoms_ai_agent.core.llm.action import Action, ActionType


# ---------------------------------------------------------------------------
# Unit tests (mocked OpenAI SDK)
# ---------------------------------------------------------------------------


class TestChatResult:
    def test_non_structured(self):
        result = ChatResult(content="你好，今天天气不错")
        assert result.content == "你好，今天天气不错"
        assert result.action is None
        assert result.is_structured is False

    def test_structured(self):
        action = Action(type=ActionType.EXIT)
        result = ChatResult(content='{"action": "exit"}', action=action)
        assert result.is_structured is True
        assert result.action is action
        assert result.action.type == ActionType.EXIT

    def test_content_always_raw_text(self):
        """content is always the original LLM response, even when structured."""
        raw = '{"action": "switch", "target": "recipe"}'
        action = Action(type=ActionType.SWITCH, target="recipe")
        result = ChatResult(content=raw, action=action)
        assert result.content == raw


class TestLLMClientInit:
    def test_creates_openai_client(self, monkeypatch):
        """Verify LLMClient initializes without error when env vars are set."""
        monkeypatch.setenv(ENV_LLM_BASE_URL, "http://localhost:11434/v1")
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "false")
        monkeypatch.setenv(ENV_LLM_MODEL, "qwen2.5:7b")
        config = LLMConfig.from_env()
        client = LLMClient(config, timeout=10, max_retries=1)
        assert client._config is config
        # OpenAI SDK normalizes base_url by appending a trailing /
        assert str(client._client.base_url).rstrip("/") == "http://localhost:11434/v1"

    def test_api_key_placeholder_for_no_auth(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_BASE_URL, "http://localhost:11434/v1")
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "false")
        monkeypatch.setenv(ENV_LLM_MODEL, "qwen2.5:7b")
        config = LLMConfig.from_env()
        client = LLMClient(config)
        assert client._client.api_key == "ollama"

    def test_api_key_from_config_when_auth_enabled(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_BASE_URL, "https://api.deepseek.com/v1")
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "true")
        monkeypatch.setenv(ENV_LLM_API_KEY, "sk-test-key")
        monkeypatch.setenv(ENV_LLM_MODEL, "deepseek-chat")
        config = LLMConfig.from_env()
        client = LLMClient(config)
        assert client._client.api_key == "sk-test-key"


class TestLLMClientChat:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_BASE_URL, "http://localhost:11434/v1")
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "false")
        monkeypatch.setenv(ENV_LLM_MODEL, "qwen2.5:7b")
        config = LLMConfig.from_env()
        return LLMClient(config)

    def test_chat_returns_chat_result(self, client):
        mock_msg = MagicMock()
        mock_msg.content = "你好！"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(client._client.chat.completions, "create", return_value=mock_response):
            result = client.chat([{"role": "user", "content": "你好"}])
            assert isinstance(result, ChatResult)
            assert result.content == "你好！"
            assert result.action is None  # plain text, no JSON

    def test_chat_parses_structured_action(self, client):
        mock_msg = MagicMock()
        mock_msg.content = '{"action": "exit"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(client._client.chat.completions, "create", return_value=mock_response):
            result = client.chat([{"role": "user", "content": "再见"}])
            assert result.is_structured
            assert result.action.type == ActionType.EXIT

    def test_chat_respects_temperature(self, client):
        mock_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "ok"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response.choices = [mock_choice]

        with patch.object(client._client.chat.completions, "create", return_value=mock_response) as mock_create:
            client.chat([{"role": "user", "content": "hi"}], temperature=0.8)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.8

    def test_chat_raises_llm_error_on_failure(self, client):
        with patch.object(
            client._client.chat.completions,
            "create",
            side_effect=Exception("Connection refused"),
        ):
            with pytest.raises(LLMError, match="Connection refused"):
                client.chat([{"role": "user", "content": "hi"}])

    def test_chat_json_mode_false_by_default(self, client):
        """When json_mode is not set, response_format must NOT be in the API call."""
        mock_msg = MagicMock()
        mock_msg.content = "ok"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(client._client.chat.completions, "create", return_value=mock_response) as mock_create:
            client.chat([{"role": "user", "content": "hi"}])
            call_kwargs = mock_create.call_args.kwargs
            assert "response_format" not in call_kwargs

    def test_chat_json_mode_true_sets_response_format(self, client):
        """When json_mode=True, response_format={'type': 'json_object'} must be passed."""
        mock_msg = MagicMock()
        mock_msg.content = '{"action": "exit"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(client._client.chat.completions, "create", return_value=mock_response) as mock_create:
            client.chat([{"role": "user", "content": "输出json"}], json_mode=True)
            call_kwargs = mock_create.call_args.kwargs
            assert "response_format" in call_kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_chat_passes_messages_through(self, client):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        mock_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "Hi!"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response.choices = [mock_choice]

        with patch.object(client._client.chat.completions, "create", return_value=mock_response) as mock_create:
            client.chat(messages)
            assert mock_create.call_args.kwargs["messages"] == messages


class TestLLMClientEmbed:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_BASE_URL, "http://localhost:11434/v1")
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "false")
        monkeypatch.setenv(ENV_LLM_MODEL, "qwen2.5:7b")
        config = LLMConfig.from_env()
        return LLMClient(config)

    def test_embed_returns_float_list(self, client):
        mock_data = MagicMock()
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_data]

        with patch.object(client._client.embeddings, "create", return_value=mock_response):
            result = client.embed("测试")
            assert result == [0.1, 0.2, 0.3]

    def test_embed_raises_llm_error_on_failure(self, client):
        with patch.object(
            client._client.embeddings,
            "create",
            side_effect=Exception("Timeout"),
        ):
            with pytest.raises(LLMError, match="Timeout"):
                client.embed("test")

    def test_embed_uses_embed_model(self, client):
        client._config.embed_model = "text-embedding-3-small"
        mock_data = MagicMock()
        mock_data.embedding = [1.0]
        mock_response = MagicMock()
        mock_response.data = [mock_data]

        with patch.object(client._client.embeddings, "create", return_value=mock_response) as mock_create:
            client.embed("text")
            assert mock_create.call_args.kwargs["model"] == "text-embedding-3-small"


class TestLLMClientEmbedBatch:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_BASE_URL, "http://localhost:11434/v1")
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "false")
        monkeypatch.setenv(ENV_LLM_MODEL, "qwen2.5:7b")
        config = LLMConfig.from_env()
        return LLMClient(config)

    def test_embed_batch_returns_list_of_lists(self, client):
        mock_data_0 = MagicMock()
        mock_data_0.index = 0
        mock_data_0.embedding = [0.1, 0.2]

        mock_data_1 = MagicMock()
        mock_data_1.index = 1
        mock_data_1.embedding = [0.3, 0.4]

        mock_response = MagicMock()
        mock_response.data = [mock_data_0, mock_data_1]

        with patch.object(client._client.embeddings, "create", return_value=mock_response):
            result = client.embed_batch(["text one", "text two"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_batch_preserves_order(self, client):
        """Results must be ordered by index, not insertion order."""
        mock_data_1 = MagicMock()
        mock_data_1.index = 1
        mock_data_1.embedding = [0.3, 0.4]

        mock_data_0 = MagicMock()
        mock_data_0.index = 0
        mock_data_0.embedding = [0.1, 0.2]

        mock_response = MagicMock()
        mock_response.data = [mock_data_1, mock_data_0]  # reversed!

        with patch.object(client._client.embeddings, "create", return_value=mock_response):
            result = client.embed_batch(["text one", "text two"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_batch_raises_llm_error(self, client):
        with patch.object(
            client._client.embeddings,
            "create",
            side_effect=Exception("API error"),
        ):
            with pytest.raises(LLMError, match="API error"):
                client.embed_batch(["a", "b"])


class TestLLMError:
    def test_string_message(self):
        err = LLMError("something went wrong")
        assert str(err) == "something went wrong"
        assert err.original is None

    def test_wraps_original_exception(self):
        original = ValueError("bad value")
        err = LLMError("failed", original=original)
        assert err.original is original
        assert "failed" in str(err)


