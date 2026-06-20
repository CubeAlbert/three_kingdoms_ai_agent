"""Tests for agents.base — BaseAgent ABC, AgentContext, and AgentResult."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from three_kingdoms_ai_agent.agents.base import (
    AgentContext,
    AgentResult,
    BaseAgent,
    _JSON_FALLBACK_SUFFIX,
)
from three_kingdoms_ai_agent.core.llm.client import ChatResult, LLMClient
from three_kingdoms_ai_agent.core.memory.base import MemoryManager


# =========================================================================
# AgentContext
# =========================================================================


class TestAgentContext:
    def test_construction_minimal(self):
        ctx = AgentContext(
            user_message="吃什么",
            sub_type="吃什么",
            matched_meme="是啊，吃什么",
            history=[],
            llm=MagicMock(spec=LLMClient),
            memory=MagicMock(spec=MemoryManager),
        )
        assert ctx.user_message == "吃什么"
        assert ctx.sub_type == "吃什么"
        assert ctx.matched_meme == "是啊，吃什么"
        assert ctx.history == []

    def test_construction_with_history(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        ctx = AgentContext(
            user_message="吃什么",
            sub_type="吃什么",
            matched_meme="吃什么",
            history=history,
            llm=MagicMock(spec=LLMClient),
            memory=MagicMock(spec=MemoryManager),
        )
        assert len(ctx.history) == 2
        assert ctx.history == history


# =========================================================================
# AgentResult
# =========================================================================


class TestAgentResult:
    def test_ok_when_data_present(self):
        result = AgentResult(
            agent_id="test_agent",
            sub_type="test_sub",
            data={"key": "value"},
            raw_content='{"key": "value"}',
        )
        assert result.is_ok is True

    def test_not_ok_when_data_is_none(self):
        result = AgentResult(
            agent_id="test_agent",
            sub_type="test_sub",
            data=None,
            raw_content="not json",
        )
        assert result.is_ok is False

    def test_raw_content_always_preserved(self):
        raw = "some raw LLM output"
        result = AgentResult(
            agent_id="a", sub_type="b", data=None, raw_content=raw
        )
        assert result.raw_content == raw

    def test_fields_match_constructor(self):
        result = AgentResult(
            agent_id="recipe_agent",
            sub_type="吃什么",
            data={"name": "红烧肉"},
            raw_content='{"name": "红烧肉"}',
        )
        assert result.agent_id == "recipe_agent"
        assert result.sub_type == "吃什么"
        assert result.data == {"name": "红烧肉"}


# =========================================================================
# BaseAgent ABC
# =========================================================================


class TestBaseAgentABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore[abstract]

    def test_concrete_subclass_can_instantiate(self):
        class MyAgent(BaseAgent):
            name = "my_agent"
            system_prompt = "You are helpful. Respond in json."
            sub_type_prompts = {"a": "sub prompt a"}

            def parse_result(self, raw_content, ctx):
                return AgentResult(
                    agent_id=self.name,
                    sub_type=ctx.sub_type,
                    data={"raw": raw_content},
                    raw_content=raw_content,
                )

        agent = MyAgent()
        assert agent.name == "my_agent"
        assert issubclass(MyAgent, BaseAgent)


# =========================================================================
# _build_messages
# =========================================================================


class TestBuildMessages:
    """Tests for BaseAgent._build_messages — three-layer assembly."""

    @staticmethod
    def _make_agent():
        class TestAgent(BaseAgent):
            name = "test"
            system_prompt = "You are a test agent. Output json."
            sub_type_prompts = {
                "eat": "Recommend food.",
                "drink": "Recommend drinks.",
            }

            def parse_result(self, raw_content, ctx):
                return AgentResult(
                    agent_id=self.name,
                    sub_type=ctx.sub_type,
                    data={"raw": raw_content},
                    raw_content=raw_content,
                )

        return TestAgent()

    @staticmethod
    def _make_ctx(sub_type="eat", user_message="吃什么", history=None):
        return AgentContext(
            user_message=user_message,
            sub_type=sub_type,
            matched_meme="吃什么",
            history=history or [],
            llm=MagicMock(spec=LLMClient),
            memory=MagicMock(spec=MemoryManager),
        )

    def test_structure_three_layers(self):
        agent = self._make_agent()
        ctx = self._make_ctx()
        messages = agent._build_messages(ctx)

        # Layer 1: system
        assert messages[0]["role"] == "system"
        # Layer 3: user (last)
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "吃什么"

    def test_system_prompt_includes_sub_type_prompt(self):
        agent = self._make_agent()
        ctx = self._make_ctx(sub_type="eat")
        messages = agent._build_messages(ctx)

        system_content = messages[0]["content"]
        assert "You are a test agent" in system_content
        assert "Recommend food." in system_content

    def test_system_prompt_selects_correct_sub_type(self):
        agent = self._make_agent()
        ctx = self._make_ctx(sub_type="drink")
        messages = agent._build_messages(ctx)

        system_content = messages[0]["content"]
        assert "Recommend drinks." in system_content
        assert "Recommend food." not in system_content

    def test_history_sandwiched_between_system_and_user(self):
        agent = self._make_agent()
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        ctx = self._make_ctx(history=history)
        messages = agent._build_messages(ctx)

        assert messages[0]["role"] == "system"
        assert messages[1] == history[0]
        assert messages[2] == history[1]
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "吃什么"

    def test_missing_sub_type_uses_empty_string(self):
        agent = self._make_agent()
        ctx = self._make_ctx(sub_type="nonexistent")
        messages = agent._build_messages(ctx)

        system_content = messages[0]["content"]
        # Should still contain system_prompt but no sub_type prompt
        assert "You are a test agent" in system_content
        # JSON fallback should NOT trigger because system_prompt has "json"
        assert _JSON_FALLBACK_SUFFIX not in system_content

    def test_no_history_produces_two_messages(self):
        agent = self._make_agent()
        ctx = self._make_ctx(history=[])
        messages = agent._build_messages(ctx)

        assert len(messages) == 2  # system + user only


# =========================================================================
# _assemble_system_prompt
# =========================================================================


def _make_minimal_agent(system_prompt="Role: chef. Output json."):
    """Create a minimal concrete BaseAgent subclass for testing.

    Uses ``type()`` so the *system_prompt* parameter is injected as a class
    attribute — a nested ``class`` statement cannot see enclosing scope
    variables for class-body assignments in Python.
    """

    def parse_result(self, raw_content, ctx):
        return AgentResult(
            agent_id=self.name,
            sub_type=ctx.sub_type,
            data={"raw": raw_content},
            raw_content=raw_content,
        )

    MinimalAgent = type(
        "MinimalAgent",
        (BaseAgent,),
        {
            "name": "minimal",
            "system_prompt": system_prompt,
            "parse_result": parse_result,
        },
    )
    return MinimalAgent()


class TestAssembleSystemPrompt:

    def test_combines_system_and_sub_prompt(self):
        agent = _make_minimal_agent("Role: chef. Output json.")
        result = agent._assemble_system_prompt("Recommend food.")
        assert "Role: chef. Output json." in result
        assert "Recommend food." in result

    def test_no_sub_prompt_omits_separator(self):
        agent = _make_minimal_agent("Role: chef. Output json.")
        result = agent._assemble_system_prompt("")
        assert result == "Role: chef. Output json."

    def test_appends_json_fallback_when_missing(self):
        agent = _make_minimal_agent("Role: chef.")  # no "json" word
        result = agent._assemble_system_prompt("")
        assert _JSON_FALLBACK_SUFFIX in result
        assert result.endswith(_JSON_FALLBACK_SUFFIX)

    def test_does_not_append_fallback_when_json_present(self):
        agent = _make_minimal_agent("You are a chef. Respond in JSON format.")
        result = agent._assemble_system_prompt("")
        assert _JSON_FALLBACK_SUFFIX not in result

    def test_json_in_sub_prompt_also_prevents_fallback(self):
        agent = _make_minimal_agent("Role: chef.")  # no json in system
        result = agent._assemble_system_prompt(
            "I need you to output valid json."
        )
        assert _JSON_FALLBACK_SUFFIX not in result


# =========================================================================
# handle() integration
# =========================================================================


class TestHandle:
    """Integration tests for BaseAgent.handle() with mocked LLMClient."""

    @staticmethod
    def _make_agent_and_ctx(sub_type="eat", llm_response='{"dish": "noodles"}'):
        class TestAgent(BaseAgent):
            name = "test"
            system_prompt = "You are a test agent. Output json."
            sub_type_prompts = {"eat": "Recommend food."}

            def parse_result(self, raw_content, ctx):
                try:
                    data = json.loads(raw_content)
                except (json.JSONDecodeError, ValueError):
                    data = None
                return AgentResult(
                    agent_id=self.name,
                    sub_type=ctx.sub_type,
                    data=data,
                    raw_content=raw_content,
                )

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat.return_value = ChatResult(
            content=llm_response,
            action=None,  # sub-agents handle their own parsing
        )

        ctx = AgentContext(
            user_message="吃什么",
            sub_type=sub_type,
            matched_meme="是啊，吃什么",
            history=[],
            llm=mock_llm,
            memory=MagicMock(spec=MemoryManager),
        )

        return TestAgent(), ctx, mock_llm

    def test_calls_llm_with_json_mode_true(self):
        agent, ctx, mock_llm = self._make_agent_and_ctx()
        agent.handle(ctx)

        mock_llm.chat.assert_called_once()
        _, kwargs = mock_llm.chat.call_args
        assert kwargs["json_mode"] is True

    def test_passes_temperature(self):
        agent, ctx, mock_llm = self._make_agent_and_ctx()
        agent.temperature = 0.5
        agent.handle(ctx)

        _, kwargs = mock_llm.chat.call_args
        assert kwargs["temperature"] == 0.5

    def test_returns_agent_result_with_parsed_data(self):
        agent, ctx, _ = self._make_agent_and_ctx(
            llm_response='{"dish": "红烧肉"}'
        )
        result = agent.handle(ctx)

        assert isinstance(result, AgentResult)
        assert result.agent_id == "test"
        assert result.sub_type == "eat"
        assert result.is_ok is True
        assert result.data == {"dish": "红烧肉"}
        assert result.raw_content == '{"dish": "红烧肉"}'

    def test_returns_agent_result_with_none_data_on_parse_failure(self):
        agent, ctx, _ = self._make_agent_and_ctx(
            llm_response="not valid json at all"
        )
        result = agent.handle(ctx)

        assert isinstance(result, AgentResult)
        assert result.is_ok is False
        assert result.data is None
        assert result.raw_content == "not valid json at all"

    def test_build_messages_structure_in_handle(self):
        agent, ctx, mock_llm = self._make_agent_and_ctx(
            sub_type="eat", llm_response='{"dish": "test"}'
        )
        agent.handle(ctx)

        messages = mock_llm.chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert "You are a test agent" in messages[0]["content"]
        assert "Recommend food." in messages[0]["content"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "吃什么"
