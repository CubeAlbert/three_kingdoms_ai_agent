"""Tests for core.prompt_loader."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

from three_kingdoms_ai_agent.core.prompt_loader import (
    AgentPrompt,
    OrchestratorPrompt,
    PromptLoader,
    load_agent_prompt,
    load_orchestrator_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_prompts_dir():
    """Create a temporary prompts/ directory with sample YAML files."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        (root / "agents").mkdir()

        (root / "agents" / "recipe.yaml").write_text(
            dedent(
                """\
                agent_id: recipe
                system_prompt: "你是一个美食助手"
                sub_type_prompts:
                  eat: "推荐食物"
                  drink: "推荐饮品"
                """
            ),
            encoding="utf-8",
        )

        (root / "orchestrator.yaml").write_text(
            dedent(
                """\
                persona: "新三国bot"
                chat_rules: "保持幽默"
                integration_rules: "确定性拼装"
                """
            ),
            encoding="utf-8",
        )

        yield root


@pytest.fixture
def loader(tmp_prompts_dir):
    """Return a PromptLoader pointed at the temp prompts directory."""
    return PromptLoader(tmp_prompts_dir)


# ---------------------------------------------------------------------------
# Agent prompt loading
# ---------------------------------------------------------------------------


class TestLoadAgent:
    def test_loads_agent_id(self, loader):
        prompt = loader.load_agent("recipe")
        assert prompt.agent_id == "recipe"

    def test_loads_system_prompt(self, loader):
        prompt = loader.load_agent("recipe")
        assert "美食助手" in prompt.system_prompt

    def test_loads_sub_type_prompts(self, loader):
        prompt = loader.load_agent("recipe")
        assert prompt.sub_type_prompts == {"eat": "推荐食物", "drink": "推荐饮品"}

    def test_get_prompt_returns_existing(self, loader):
        prompt = loader.load_agent("recipe")
        assert prompt.get_prompt("eat") == "推荐食物"

    def test_get_prompt_returns_none_for_missing(self, loader):
        prompt = loader.load_agent("recipe")
        assert prompt.get_prompt("nonexistent") is None

    def test_missing_file_returns_empty_prompt(self, loader):
        prompt = loader.load_agent("ghost_agent")
        assert prompt.agent_id == "ghost_agent"
        assert prompt.system_prompt == ""
        assert prompt.sub_type_prompts == {}


# ---------------------------------------------------------------------------
# Orchestrator prompt loading
# ---------------------------------------------------------------------------


class TestLoadOrchestrator:
    def test_loads_persona(self, loader):
        prompt = loader.load_orchestrator()
        assert "新三国bot" in prompt.persona

    def test_loads_chat_rules(self, loader):
        prompt = loader.load_orchestrator()
        assert "保持幽默" in prompt.chat_rules

    def test_loads_integration_rules(self, loader):
        prompt = loader.load_orchestrator()
        assert "确定性拼装" in prompt.integration_rules

    def test_missing_orchestrator_file_returns_empty(self, tmp_path):
        loader = PromptLoader(tmp_path)
        prompt = loader.load_orchestrator()
        assert prompt.persona == ""
        assert prompt.chat_rules == ""
        assert prompt.integration_rules == ""


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


class TestCache:
    def test_same_agent_returns_cached_instance(self, loader):
        a = loader.load_agent("recipe")
        b = loader.load_agent("recipe")
        assert a is b

    def test_orchestrator_is_cached(self, loader):
        a = loader.load_orchestrator()
        b = loader.load_orchestrator()
        assert a is b

    def test_clear_cache_discards_entries(self, loader):
        a = loader.load_agent("recipe")
        loader.clear_cache()
        b = loader.load_agent("recipe")
        assert a is not b
        # content should still match
        assert a.system_prompt == b.system_prompt

    def test_cache_separates_agent_and_orchestrator(self, loader):
        agent = loader.load_agent("recipe")
        orch = loader.load_orchestrator()
        assert not isinstance(agent, OrchestratorPrompt)
        assert not isinstance(orch, AgentPrompt)


# ---------------------------------------------------------------------------
# ENV_VAR substitution
# ---------------------------------------------------------------------------


class TestEnvSubstitution:
    def test_replaces_env_var(self, tmp_path):
        (tmp_path / "agents").mkdir()
        (tmp_path / "agents" / "test.yaml").write_text(
            'agent_id: test\nsystem_prompt: "key=${MY_KEY}"\n',
            encoding="utf-8",
        )
        os.environ["MY_KEY"] = "secret123"
        try:
            loader = PromptLoader(tmp_path)
            prompt = loader.load_agent("test")
            assert "secret123" in prompt.system_prompt
            assert "${MY_KEY}" not in prompt.system_prompt
        finally:
            del os.environ["MY_KEY"]

    def test_unknown_var_left_untouched(self, tmp_path):
        (tmp_path / "agents").mkdir()
        (tmp_path / "agents" / "test.yaml").write_text(
            'agent_id: test\nsystem_prompt: "${UNDEFINED_VAR}"\n',
            encoding="utf-8",
        )
        loader = PromptLoader(tmp_path)
        prompt = loader.load_agent("test")
        assert "${UNDEFINED_VAR}" in prompt.system_prompt

    def test_substitutes_in_sub_type_prompts(self, tmp_path):
        (tmp_path / "agents").mkdir()
        (tmp_path / "agents" / "test.yaml").write_text(
            dedent(
                """\
                agent_id: test
                system_prompt: "hi"
                sub_type_prompts:
                  a: "url=${API_URL}"
                """
            ),
            encoding="utf-8",
        )
        os.environ["API_URL"] = "https://api.example.com"
        try:
            loader = PromptLoader(tmp_path)
            prompt = loader.load_agent("test")
            assert prompt.get_prompt("a") == "url=https://api.example.com"
        finally:
            del os.environ["API_URL"]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_load_agent_prompt_one_shot(self, tmp_prompts_dir):
        prompt = load_agent_prompt("recipe", tmp_prompts_dir)
        assert isinstance(prompt, AgentPrompt)
        assert prompt.agent_id == "recipe"

    def test_load_orchestrator_prompt_one_shot(self, tmp_prompts_dir):
        prompt = load_orchestrator_prompt(tmp_prompts_dir)
        assert isinstance(prompt, OrchestratorPrompt)
        assert "新三国bot" in prompt.persona


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


class TestAgentPromptDefaults:
    def test_minimal_construction(self):
        p = AgentPrompt(agent_id="minimal", system_prompt="hello")
        assert p.sub_type_prompts == {}
        assert p.get_prompt("any") is None


class TestOrchestratorPromptDefaults:
    def test_empty_construction(self):
        p = OrchestratorPrompt()
        assert p.persona == ""
        assert p.chat_rules == ""
        assert p.integration_rules == ""
