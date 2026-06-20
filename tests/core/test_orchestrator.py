"""Tests for core.orchestrator — exit detection, templates, hit/miss paths."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from three_kingdoms_ai_agent.agents.base import AgentContext, AgentResult, BaseAgent
from three_kingdoms_ai_agent.core.channel.base import AgentResponse, Channel, Message
from three_kingdoms_ai_agent.core.llm.client import ChatResult, LLMClient
from three_kingdoms_ai_agent.core.memory.base import MemoryManager
from three_kingdoms_ai_agent.core.orchestrator import (
    CHAT_RULES,
    PERSONA,
    Orchestrator,
    _fallback_template,
    _is_exit,
    _recipe_template,
)
from three_kingdoms_ai_agent.core.rag.router import RouteResult, Router


# =========================================================================
# _is_exit
# =========================================================================


class TestIsExit:
    @pytest.mark.parametrize(
        "text",
        ["exit", "EXIT", "quit", "q", "退出", "告辞", "拜别", " 退出 "],
    )
    def test_exit_keywords(self, text):
        assert _is_exit(text) is True

    @pytest.mark.parametrize(
        "text",
        ["hello", "吃什么", "", "ex it", "quits"],
    )
    def test_non_exit_text(self, text):
        assert _is_exit(text) is False


# =========================================================================
# _recipe_template
# =========================================================================


class TestRecipeTemplate:
    def _make_result(self, sub_type="吃什么", data=None, raw=""):
        if data is None:
            data = {
                "name": "红烧肉",
                "reason": "关将军最爱",
                "description": "五花肉文火慢炖",
                "suggestion": "配黄酒一壶",
            }
        return AgentResult(
            agent_id="recipe_agent",
            sub_type=sub_type,
            data=data,
            raw_content=raw,
        )

    def test_eat_sub_type_uses_food_intro(self):
        result = self._make_result(sub_type="吃什么")
        text = _recipe_template(result)
        assert "腹中饥馑" in text
        assert "红烧肉" in text
        assert "关将军最爱" in text

    def test_drink_sub_type_uses_drink_intro(self):
        result = self._make_result(sub_type="喝什么")
        text = _recipe_template(result)
        assert "杯中物" in text
        assert "红烧肉" in text

    def test_includes_description(self):
        result = self._make_result()
        text = _recipe_template(result)
        assert "五花肉文火慢炖" in text

    def test_includes_suggestion(self):
        result = self._make_result()
        text = _recipe_template(result)
        assert "配黄酒一壶" in text

    def test_no_description_omitted(self):
        data = {"name": "白斩鸡", "reason": "清淡鲜美"}
        result = self._make_result(data=data)
        text = _recipe_template(result)
        assert "白斩鸡" in text
        # No empty lines for missing fields — just intro + suggestion
        assert "清淡鲜美" in text

    def test_no_suggestion_omitted(self):
        data = {"name": "白斩鸡", "reason": "清淡鲜美"}
        result = self._make_result(data=data)
        text = _recipe_template(result)
        assert "配" not in text  # suggestion section should be absent

    def test_none_data_shows_placeholders(self):
        result = AgentResult(
            agent_id="recipe_agent",
            sub_type="吃什么",
            data=None,
            raw_content="raw",
        )
        text = _recipe_template(result)
        assert "???" in text
        assert "天机不可泄露" in text


# =========================================================================
# _fallback_template
# =========================================================================


class TestFallbackTemplate:
    def test_with_data_renders_key_value_list(self):
        result = AgentResult(
            agent_id="unknown_agent",
            sub_type="x",
            data={"k1": "v1", "k2": "v2"},
            raw_content="raw",
        )
        text = _fallback_template(result)
        assert "k1" in text
        assert "v1" in text
        assert "k2" in text
        assert "v2" in text
        assert "探子来报" in text

    def test_without_data_renders_raw_content(self):
        result = AgentResult(
            agent_id="unknown_agent",
            sub_type="x",
            data=None,
            raw_content="LLM 直接返回的文本",
        )
        text = _fallback_template(result)
        assert "LLM 直接返回的文本" in text
        assert "军师曰" in text


# =========================================================================
# Orchestrator helpers
# =========================================================================


def _make_mock_llm(chat_return="军师回复"):
    mock = MagicMock(spec=LLMClient)
    mock.chat.return_value = ChatResult(content=chat_return, action=None)
    return mock


def _make_mock_router(route_return=None):
    mock = MagicMock(spec=Router)
    mock.route.return_value = route_return
    return mock


def _make_mock_channel(messages=None):
    """Return a mock Channel that yields *messages* then raises EOFError."""
    mock = MagicMock(spec=Channel)
    if messages is None:
        messages = ["你好"]
    # Build side_effect: one Message per input, then EOFError
    side_effects = [Message(content=m) for m in messages] + [EOFError()]
    mock.receive.side_effect = side_effects
    return mock


def _make_orchestrator(
    channel=None,
    llm=None,
    router=None,
    memory=None,
    agents=None,
    templates=None,
):
    return Orchestrator(
        channel=channel or _make_mock_channel(["退出"]),
        llm=llm or _make_mock_llm(),
        router=router or _make_mock_router(),
        memory=memory or MagicMock(spec=MemoryManager),
        agents=agents or {},
        templates=templates,
    )


# =========================================================================
# _render_hit
# =========================================================================


class TestRenderHit:
    def test_uses_registered_template(self):
        orch = _make_orchestrator()

        result = AgentResult(
            agent_id="recipe_agent",
            sub_type="吃什么",
            data={
                "name": "测试菜",
                "reason": "测试理由",
                "description": "测试描述",
                "suggestion": "测试建议",
            },
            raw_content="{}",
        )
        text = orch._render_hit(result)
        assert "测试菜" in text
        assert "腹中饥馑" in text  # from _recipe_template eat intro

    def test_falls_back_for_unregistered_agent(self):
        orch = _make_orchestrator()

        result = AgentResult(
            agent_id="nonexistent",
            sub_type="x",
            data={"key": "value"},
            raw_content="{}",
        )
        text = orch._render_hit(result)
        assert "探子来报" in text  # from _fallback_template
        assert "key" in text

    def test_falls_back_when_template_raises(self):
        def _broken(result):
            raise RuntimeError("boom")

        orch = _make_orchestrator(
            templates={"recipe_agent": _broken}
        )

        result = AgentResult(
            agent_id="recipe_agent",
            sub_type="吃什么",
            data={"name": "红烧肉"},
            raw_content="{}",
        )
        text = orch._render_hit(result)
        # Should fall through to _fallback_template
        assert "探子来报" in text

    def test_caller_template_overrides_default(self):
        def _custom(result):
            return "自定义模板"

        orch = _make_orchestrator(
            templates={"recipe_agent": _custom}
        )

        result = AgentResult(
            agent_id="recipe_agent",
            sub_type="吃什么",
            data={},
            raw_content="",
        )
        assert orch._render_hit(result) == "自定义模板"


# =========================================================================
# _handle_hit
# =========================================================================


class TestHandleHit:
    def test_routes_to_registered_agent(self):
        class FakeAgent(BaseAgent):
            name = "recipe_agent"
            system_prompt = "test. Output json."
            sub_type_prompts = {"吃什么": "test prompt"}

            def parse_result(self, raw_content, ctx):
                return AgentResult(
                    agent_id=self.name,
                    sub_type=ctx.sub_type,
                    data={"name": "红烧肉"},
                    raw_content=raw_content,
                )

        mock_llm = _make_mock_llm('{"name": "红烧肉"}')
        orch = _make_orchestrator(
            llm=mock_llm,
            agents={"recipe_agent": FakeAgent()},
        )

        route = RouteResult(
            agent_id="recipe_agent",
            sub_type="吃什么",
            meme_text="是啊，吃什么",
            similarity=0.9,
        )
        text = orch._handle_hit(route, "吃什么")

        assert "红烧肉" in text
        assert "腹中饥馑" in text

    def test_falls_back_to_chat_when_agent_not_registered(self):
        mock_llm = _make_mock_llm("军师闲聊回复")
        mock_memory = MagicMock(spec=MemoryManager)
        mock_memory.get_context.return_value = [
            {"role": "user", "content": "你好"}
        ]

        orch = _make_orchestrator(
            llm=mock_llm,
            memory=mock_memory,
            agents={},  # empty registry
        )

        route = RouteResult(
            agent_id="recipe_agent",
            sub_type="吃什么",
            meme_text="吃什么",
            similarity=0.9,
        )
        text = orch._handle_hit(route, "吃什么")

        # Should have fallen back to chat
        assert text == "军师闲聊回复"
        mock_llm.chat.assert_called_once()

    def test_falls_back_when_agent_raises(self):
        class CrashAgent(BaseAgent):
            name = "crash_agent"
            system_prompt = "test. Output json."
            sub_type_prompts = {"x": "prompt"}

            def parse_result(self, raw_content, ctx):
                return AgentResult(
                    agent_id=self.name,
                    sub_type=ctx.sub_type,
                    data=None,
                    raw_content=raw_content,
                )

            def handle(self, ctx):
                raise RuntimeError("boom")

        mock_llm = _make_mock_llm("fallback chat")
        mock_memory = MagicMock(spec=MemoryManager)
        mock_memory.get_context.return_value = []

        orch = _make_orchestrator(
            llm=mock_llm,
            memory=mock_memory,
            agents={"crash_agent": CrashAgent()},
        )

        route = RouteResult(
            agent_id="crash_agent",
            sub_type="x",
            meme_text="test",
            similarity=0.9,
        )
        text = orch._handle_hit(route, "test")
        assert text == "fallback chat"


# =========================================================================
# _handle_miss
# =========================================================================


class TestHandleMiss:
    def test_uses_persona_and_chat_rules_in_system_prompt(self):
        mock_llm = _make_mock_llm("闲聊回复")
        mock_memory = MagicMock(spec=MemoryManager)
        mock_memory.get_context.return_value = [
            {"role": "user", "content": "你好"}
        ]

        orch = _make_orchestrator(llm=mock_llm, memory=mock_memory)
        text = orch._handle_miss("今天天气不错")

        assert text == "闲聊回复"

        # Verify the system prompt was assembled correctly
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert PERSONA in system_msg["content"]
        assert CHAT_RULES in system_msg["content"]

    def test_includes_history(self):
        mock_llm = _make_mock_llm()
        mock_memory = MagicMock(spec=MemoryManager)
        history = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        mock_memory.get_context.return_value = history

        orch = _make_orchestrator(llm=mock_llm, memory=mock_memory)
        orch._handle_miss("Q2")

        messages = mock_llm.chat.call_args[0][0]
        # System + Q1 + A1 + Q2 = 4 messages
        assert len(messages) == 4
        assert messages[1] == history[0]
        assert messages[2] == history[1]
        assert messages[3] == {"role": "user", "content": "Q2"}

    def test_json_mode_is_false(self):
        mock_llm = _make_mock_llm()
        mock_memory = MagicMock(spec=MemoryManager)
        mock_memory.get_context.return_value = []

        orch = _make_orchestrator(llm=mock_llm, memory=mock_memory)
        orch._handle_miss("hello")

        _, kwargs = mock_llm.chat.call_args
        assert kwargs["json_mode"] is False

    def test_returns_canned_response_on_llm_failure(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat.side_effect = RuntimeError("API down")
        mock_memory = MagicMock(spec=MemoryManager)
        mock_memory.get_context.return_value = []

        orch = _make_orchestrator(llm=mock_llm, memory=mock_memory)
        text = orch._handle_miss("hello")

        assert "再思量" in text


# =========================================================================
# run() integration
# =========================================================================


class TestRun:
    def test_exits_on_eof(self):
        channel = _make_mock_channel([])  # EOF immediately
        orch = _make_orchestrator(channel=channel)
        orch.run()  # should not hang or raise

    def test_exits_on_keyword(self):
        channel = _make_mock_channel(["退出"])
        orch = _make_orchestrator(channel=channel)
        orch.run()

    def test_skips_empty_input(self):
        channel = MagicMock(spec=Channel)
        channel.receive.side_effect = [
            Message(content=""),     # empty → skipped
            Message(content="  "),   # whitespace → skipped
            Message(content="退出"),  # exit
        ]
        orch = _make_orchestrator(channel=channel)
        orch.run()
        # Should process without error — empty/whitespace skipped

    def test_routes_hit_to_agent(self):
        """End-to-end: user says '吃什么' → RAG hit → recipe agent → template."""
        class FakeAgent(BaseAgent):
            name = "recipe_agent"
            system_prompt = "test json. Output json."
            sub_type_prompts = {"吃什么": "prompt"}

            def parse_result(self, raw_content, ctx):
                import json
                data = json.loads(raw_content)
                return AgentResult(
                    agent_id=self.name,
                    sub_type=ctx.sub_type,
                    data=data,
                    raw_content=raw_content,
                )

        channel = MagicMock(spec=Channel)
        channel.receive.side_effect = [
            Message(content="吃什么"),
            Message(content="退出"),
        ]

        router = _make_mock_router(
            RouteResult("recipe_agent", "吃什么", "是啊，吃什么", 0.9)
        )
        llm = _make_mock_llm('{"name":"红烧肉","reason":"好吃","description":"香","suggestion":"配酒"}')
        memory = MagicMock(spec=MemoryManager)
        memory.get_context.return_value = []

        orch = _make_orchestrator(
            channel=channel,
            llm=llm,
            router=router,
            memory=memory,
            agents={"recipe_agent": FakeAgent()},
        )
        orch.run()

        # Verify the agent response was sent
        send_calls = channel.send.call_args_list
        # First call: welcome message
        assert "军师已至" in send_calls[0][0][0].content
        # Second call: recipe result
        assert "红烧肉" in send_calls[1][0][0].content
        # Third call: exit farewell
        assert "后会有期" in send_calls[2][0][0].content

    def test_routes_miss_to_chat(self):
        """End-to-end: user says 'hello' → RAG miss → chat LLM."""
        channel = MagicMock(spec=Channel)
        channel.receive.side_effect = [
            Message(content="hello"),
            Message(content="退出"),
        ]

        router = _make_mock_router(None)  # always miss
        llm = _make_mock_llm("军师曰：来者何人？")
        memory = MagicMock(spec=MemoryManager)
        memory.get_context.return_value = []

        orch = _make_orchestrator(
            channel=channel, llm=llm, router=router, memory=memory
        )
        orch.run()

        send_calls = channel.send.call_args_list
        # Second call should be the chat response
        assert "来者何人" in send_calls[1][0][0].content

    def test_memory_adds_user_and_assistant(self):
        channel = MagicMock(spec=Channel)
        channel.receive.side_effect = [
            Message(content="hello"),
            Message(content="退出"),
        ]

        router = _make_mock_router(None)
        llm = _make_mock_llm("reply")
        memory = MagicMock(spec=MemoryManager)
        memory.get_context.return_value = []

        orch = _make_orchestrator(
            channel=channel, llm=llm, router=router, memory=memory
        )
        orch.run()

        # Memory.add should be called for user + assistant
        add_calls = memory.add.call_args_list
        assert len(add_calls) == 2
        assert add_calls[0][0] == ("user", "hello")
        assert add_calls[1][0] == ("assistant", "reply")
