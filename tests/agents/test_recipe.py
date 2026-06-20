"""Tests for agents.recipe — RecipeAgent with 吃什么 / 喝什么 sub-types."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from three_kingdoms_ai_agent.agents.base import AgentContext, AgentResult, BaseAgent
from three_kingdoms_ai_agent.agents.recipe import RecipeAgent
from three_kingdoms_ai_agent.core.llm.client import ChatResult, LLMClient
from three_kingdoms_ai_agent.core.memory.base import MemoryManager


# =========================================================================
# Agent attributes
# =========================================================================


class TestRecipeAgentAttributes:
    def test_is_base_agent_subclass(self):
        assert issubclass(RecipeAgent, BaseAgent)

    def test_name(self):
        assert RecipeAgent.name == "recipe_agent"

    def test_description(self):
        assert len(RecipeAgent.description) > 0

    def test_system_prompt_contains_json(self):
        """json_mode requires the word 'json' in the prompt."""
        assert "json" in RecipeAgent.system_prompt.lower()

    def test_system_prompt_contains_output_schema(self):
        """The prompt should show the expected JSON field structure."""
        sp = RecipeAgent.system_prompt
        assert "name" in sp
        assert "reason" in sp
        assert "description" in sp
        assert "suggestion" in sp

    def test_sub_type_prompts_has_required_keys(self):
        prompts = RecipeAgent.sub_type_prompts
        assert "吃什么" in prompts
        assert "喝什么" in prompts
        assert len(prompts[("吃什么")]) > 0
        assert len(prompts[("喝什么")]) > 0


# =========================================================================
# _build_messages
# =========================================================================


class TestRecipeAgentBuildMessages:
    @staticmethod
    def _make_ctx(sub_type="吃什么"):
        return AgentContext(
            user_message="是啊，吃什么",
            sub_type=sub_type,
            matched_meme="是啊，吃什么",
            history=[],
            llm=MagicMock(spec=LLMClient),
            memory=MagicMock(spec=MemoryManager),
        )

    def test_eat_sub_type_selects_correct_prompt(self):
        agent = RecipeAgent()
        ctx = self._make_ctx(sub_type="吃什么")
        messages = agent._build_messages(ctx)

        system = messages[0]["content"]
        assert "今天吃什么" in system

    def test_drink_sub_type_selects_correct_prompt(self):
        agent = RecipeAgent()
        ctx = self._make_ctx(sub_type="喝什么")
        messages = agent._build_messages(ctx)

        system = messages[0]["content"]
        assert "当浮一大白" in system or "饮酒" in system

    def test_system_prompt_contains_agent_persona(self):
        agent = RecipeAgent()
        ctx = self._make_ctx()
        messages = agent._build_messages(ctx)

        system = messages[0]["content"]
        assert "随军厨子" in system or "三国" in system

    def test_no_json_fallback_appended(self):
        """RecipeAgent's system_prompt already contains 'json', so the
        fallback suffix should NOT be appended."""
        agent = RecipeAgent()
        ctx = self._make_ctx()
        messages = agent._build_messages(ctx)

        system = messages[0]["content"]
        from three_kingdoms_ai_agent.agents.base import _JSON_FALLBACK_SUFFIX

        assert _JSON_FALLBACK_SUFFIX not in system


# =========================================================================
# parse_result
# =========================================================================


class TestRecipeAgentParseResult:
    @staticmethod
    def _make_ctx(sub_type="吃什么"):
        return AgentContext(
            user_message="吃什么",
            sub_type=sub_type,
            matched_meme="吃什么",
            history=[],
            llm=MagicMock(spec=LLMClient),
            memory=MagicMock(spec=MemoryManager),
        )

    def test_happy_path_parses_valid_json(self):
        agent = RecipeAgent()
        raw = json.dumps(
            {
                "name": "红烧肉",
                "reason": "关将军最爱，肥而不腻",
                "description": "选用上等五花肉，文火慢炖两个时辰",
                "suggestion": "配绍兴黄酒，当浮一大白",
            },
            ensure_ascii=False,
        )
        ctx = self._make_ctx(sub_type="吃什么")

        result = agent.parse_result(raw, ctx)

        assert isinstance(result, AgentResult)
        assert result.agent_id == "recipe_agent"
        assert result.sub_type == "吃什么"
        assert result.is_ok is True
        assert result.data["name"] == "红烧肉"
        assert result.data["reason"] == "关将军最爱，肥而不腻"
        assert result.raw_content == raw

    def test_happy_path_drink_sub_type(self):
        agent = RecipeAgent()
        raw = json.dumps(
            {
                "name": "杜康酒",
                "reason": "何以解忧，唯有杜康",
                "description": "曹操最爱之美酒，醇香四溢",
                "suggestion": "以青铜爵盛之，对月独饮",
            },
            ensure_ascii=False,
        )
        ctx = self._make_ctx(sub_type="喝什么")

        result = agent.parse_result(raw, ctx)
        assert result.is_ok is True
        assert result.sub_type == "喝什么"
        assert result.data["name"] == "杜康酒"

    def test_invalid_json_returns_none_data(self):
        agent = RecipeAgent()
        ctx = self._make_ctx()
        raw = "这不是 JSON，是随军厨子的碎碎念..."

        result = agent.parse_result(raw, ctx)

        assert result.is_ok is False
        assert result.data is None
        assert result.raw_content == raw
        assert result.agent_id == "recipe_agent"

    def test_non_dict_json_returns_none_data(self):
        agent = RecipeAgent()
        ctx = self._make_ctx()
        raw = "[1, 2, 3]"

        result = agent.parse_result(raw, ctx)

        assert result.is_ok is False
        assert result.data is None
        assert result.raw_content == raw

    def test_empty_string_returns_none_data(self):
        agent = RecipeAgent()
        ctx = self._make_ctx()

        result = agent.parse_result("", ctx)

        assert result.is_ok is False
        assert result.data is None


# =========================================================================
# handle() integration
# =========================================================================


class TestRecipeAgentHandle:
    def test_full_pipeline_returns_structured_result(self):
        agent = RecipeAgent()
        mock_llm = MagicMock(spec=LLMClient)

        response_json = json.dumps(
            {
                "name": "水煮鱼",
                "reason": "麻辣鲜香，如张飞之勇猛",
                "description": "鲜鱼切片，滚汤烫熟，麻辣入味",
                "suggestion": "配冰镇啤酒，解辣又痛快",
            },
            ensure_ascii=False,
        )
        mock_llm.chat.return_value = ChatResult(
            content=response_json, action=None
        )

        ctx = AgentContext(
            user_message="吃什么",
            sub_type="吃什么",
            matched_meme="是啊，吃什么",
            history=[],
            llm=mock_llm,
            memory=MagicMock(spec=MemoryManager),
        )

        result = agent.handle(ctx)

        assert result.is_ok is True
        assert result.data["name"] == "水煮鱼"
        assert result.agent_id == "recipe_agent"
        assert result.sub_type == "吃什么"

        # Verify LLM was called with json_mode=True
        _, kwargs = mock_llm.chat.call_args
        assert kwargs["json_mode"] is True

    def test_handle_preserves_raw_content_on_parse_failure(self):
        agent = RecipeAgent()
        mock_llm = MagicMock(spec=LLMClient)
        bad_response = "军师今日不想推荐..."
        mock_llm.chat.return_value = ChatResult(
            content=bad_response, action=None
        )

        ctx = AgentContext(
            user_message="喝什么",
            sub_type="喝什么",
            matched_meme="当浮一大白",
            history=[],
            llm=mock_llm,
            memory=MagicMock(spec=MemoryManager),
        )

        result = agent.handle(ctx)

        assert result.is_ok is False
        assert result.data is None
        assert result.raw_content == bad_response
