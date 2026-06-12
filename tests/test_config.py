"""Tests for core.config — LLMConfig, Settings, ConfigLoader, env-var reading."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

from three_kingdoms_ai_agent.core.config import (
    ENV_LLM_API_KEY,
    ENV_LLM_AUTH_ENABLED,
    ENV_LLM_BASE_URL,
    ENV_LLM_MODEL,
    ConfigLoader,
    LLMConfig,
    MemorySettings,
    RAGSettings,
    Settings,
    _substitute_dict,
    _substitute_env,
)


# ============================================================================
# LLMConfig — from_env()
# ============================================================================


class TestLLMConfigFromEnv:
    """LLM config is sourced exclusively from the four environment variables."""

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _set_all(monkeypatch):
        monkeypatch.setenv(ENV_LLM_BASE_URL, "https://api.deepseek.com/v1")
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "true")
        monkeypatch.setenv(ENV_LLM_API_KEY, "sk-test-123")
        monkeypatch.setenv(ENV_LLM_MODEL, "deepseek-chat")

    @staticmethod
    def _clear_all(monkeypatch):
        for var in (
            ENV_LLM_BASE_URL,
            ENV_LLM_AUTH_ENABLED,
            ENV_LLM_API_KEY,
            ENV_LLM_MODEL,
        ):
            monkeypatch.delenv(var, raising=False)

    # -- tests ----------------------------------------------------------------

    def test_reads_all_four_vars(self, monkeypatch):
        self._set_all(monkeypatch)
        cfg = LLMConfig.from_env()
        assert cfg.base_url == "https://api.deepseek.com/v1"
        assert cfg.auth_enabled is True
        assert cfg.api_key == "sk-test-123"
        assert cfg.model == "deepseek-chat"

    def test_defaults_when_no_vars_set(self, monkeypatch):
        self._clear_all(monkeypatch)
        cfg = LLMConfig.from_env()
        assert cfg.base_url == ""
        assert cfg.auth_enabled is True  # default: true
        assert cfg.api_key == ""
        assert cfg.model == ""

    def test_auth_disabled_via_env(self, monkeypatch):
        """LLM_AUTH_ENABLED=false → auth_enabled=False (ollama mode)."""
        self._set_all(monkeypatch)
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "false")
        cfg = LLMConfig.from_env()
        assert cfg.auth_enabled is False

    def test_auth_enabled_case_insensitive(self, monkeypatch):
        self._set_all(monkeypatch)
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "FALSE")
        cfg = LLMConfig.from_env()
        assert cfg.auth_enabled is False

    def test_auth_enabled_garbage_treated_as_false(self, monkeypatch):
        self._set_all(monkeypatch)
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "yes")  # not "true"
        cfg = LLMConfig.from_env()
        assert cfg.auth_enabled is False


# ============================================================================
# LLMConfig — validate()
# ============================================================================


class TestLLMConfigValidate:
    def test_empty_config_reports_all_issues(self, monkeypatch):
        for var in (
            ENV_LLM_BASE_URL,
            ENV_LLM_AUTH_ENABLED,
            ENV_LLM_API_KEY,
            ENV_LLM_MODEL,
        ):
            monkeypatch.delenv(var, raising=False)
        cfg = LLMConfig.from_env()
        issues = cfg.validate()
        assert len(issues) == 3  # base_url + model + (auth w/o key)
        assert any("LLM_BASE_URL" in i for i in issues)
        assert any("LLM_MODEL" in i for i in issues)
        assert any("LLM_API_KEY" in i for i in issues)

    def test_valid_config_returns_empty(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_BASE_URL, "http://localhost:11434/v1")
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "false")
        monkeypatch.setenv(ENV_LLM_MODEL, "qwen2.5:7b")
        # no API key needed when auth is disabled
        cfg = LLMConfig.from_env()
        assert cfg.validate() == []

    def test_auth_on_without_key_reports_issue(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_BASE_URL, "https://api.example.com/v1")
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "true")
        monkeypatch.delenv(ENV_LLM_API_KEY, raising=False)
        monkeypatch.setenv(ENV_LLM_MODEL, "gpt-4")
        cfg = LLMConfig.from_env()
        issues = cfg.validate()
        assert any("LLM_API_KEY" in i for i in issues)

    def test_missing_base_url_reported(self, monkeypatch):
        monkeypatch.delenv(ENV_LLM_BASE_URL, raising=False)
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "false")
        monkeypatch.setenv(ENV_LLM_MODEL, "qwen2.5:7b")
        cfg = LLMConfig.from_env()
        issues = cfg.validate()
        assert any("LLM_BASE_URL" in i for i in issues)

    def test_missing_model_reported(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_BASE_URL, "https://api.example.com/v1")
        monkeypatch.setenv(ENV_LLM_AUTH_ENABLED, "false")
        monkeypatch.delenv(ENV_LLM_MODEL, raising=False)
        cfg = LLMConfig.from_env()
        issues = cfg.validate()
        assert any("LLM_MODEL" in i for i in issues)


# ============================================================================
# ConfigLoader — load_settings()
# ============================================================================


class TestLoadSettings:
    """Settings are loaded from config/settings.yaml."""

    @pytest.fixture
    def config_dir(self) -> Path:
        """Real project config/ directory."""
        return Path("config")

    def test_loads_from_real_config(self, config_dir):
        loader = ConfigLoader(config_dir)
        settings = loader.load_settings()
        assert isinstance(settings, Settings)
        assert isinstance(settings.rag, RAGSettings)
        assert isinstance(settings.memory, MemorySettings)

    def test_rag_defaults(self, config_dir):
        settings = ConfigLoader(config_dir).load_settings()
        assert settings.rag.similarity_threshold == 0.75
        assert settings.rag.top_k == 3

    def test_memory_defaults(self, config_dir):
        settings = ConfigLoader(config_dir).load_settings()
        assert settings.memory.window_size == 10

    def test_debug_flag(self, config_dir):
        settings = ConfigLoader(config_dir).load_settings()
        assert settings.debug is False

    def test_missing_file_returns_defaults(self, tmp_path):
        loader = ConfigLoader(tmp_path)
        settings = loader.load_settings()
        assert settings.rag.similarity_threshold == 0.75
        assert settings.rag.top_k == 3
        assert settings.memory.window_size == 10
        assert settings.debug is False

    def test_custom_values_loaded(self, tmp_path):
        (tmp_path / "settings.yaml").write_text(
            dedent(
                """\
                rag:
                  similarity_threshold: 0.5
                  top_k: 5
                memory:
                  window_size: 20
                debug: true
                """
            ),
            encoding="utf-8",
        )
        settings = ConfigLoader(tmp_path).load_settings()
        assert settings.rag.similarity_threshold == 0.5
        assert settings.rag.top_k == 5
        assert settings.memory.window_size == 20
        assert settings.debug is True


# ============================================================================
# ConfigLoader — load_llm_options()
# ============================================================================


class TestLoadLLMOptions:
    @pytest.fixture
    def config_dir(self) -> Path:
        return Path("config")

    def test_loads_from_real_config(self, config_dir):
        opts = ConfigLoader(config_dir).load_llm_options()
        assert isinstance(opts, dict)
        assert "timeout" in opts
        assert "max_retries" in opts

    def test_default_values(self, config_dir):
        opts = ConfigLoader(config_dir).load_llm_options()
        assert opts["timeout"] == 60
        assert opts["max_retries"] == 3

    def test_custom_values(self, tmp_path):
        (tmp_path / "llm.yaml").write_text(
            dedent(
                """\
                timeout: 30
                max_retries: 1
                """
            ),
            encoding="utf-8",
        )
        opts = ConfigLoader(tmp_path).load_llm_options()
        assert opts["timeout"] == 30
        assert opts["max_retries"] == 1

    def test_missing_file_returns_empty(self, tmp_path):
        opts = ConfigLoader(tmp_path).load_llm_options()
        assert opts == {}


# ============================================================================
# ENV_VAR substitution in YAML values
# ============================================================================


class TestEnvSubstitutionInSettings:
    def test_substitutes_in_nested_values(self, tmp_path):
        (tmp_path / "settings.yaml").write_text(
            dedent(
                """\
                rag:
                  similarity_threshold: ${THRESHOLD}
                """
            ),
            encoding="utf-8",
        )
        os.environ["THRESHOLD"] = "0.33"
        try:
            settings = ConfigLoader(tmp_path).load_settings()
            assert settings.rag.similarity_threshold == 0.33
        finally:
            del os.environ["THRESHOLD"]

    def test_unknown_var_remains_and_raises_on_cast(self, tmp_path):
        """Unknown env vars are left as-is, so type coercion fails loudly.
        This is intentional — silent fallbacks hide misconfiguration.
        """
        (tmp_path / "settings.yaml").write_text(
            dedent(
                """\
                rag:
                  similarity_threshold: ${UNDEFINED_THRESHOLD}
                """
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="UNDEFINED_THRESHOLD"):
            ConfigLoader(tmp_path).load_settings()


# ============================================================================
# _substitute_env helper
# ============================================================================


class TestSubstituteEnv:
    def test_single_var(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "hello")
        assert _substitute_env("${MY_VAR}") == "hello"

    def test_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("A", "1")
        monkeypatch.setenv("B", "2")
        assert _substitute_env("${A}-${B}") == "1-2"

    def test_unknown_var_preserved(self):
        assert _substitute_env("${NO_SUCH_VAR}") == "${NO_SUCH_VAR}"

    def test_no_placeholders_returns_unchanged(self):
        assert _substitute_env("plain text") == "plain text"

    def test_empty_string(self):
        assert _substitute_env("") == ""


# ============================================================================
# _substitute_dict helper
# ============================================================================


class TestSubstituteDict:
    def test_flat_dict(self, monkeypatch):
        monkeypatch.setenv("X", "val")
        result = _substitute_dict({"key": "${X}"})
        assert result == {"key": "val"}

    def test_nested_dict(self, monkeypatch):
        monkeypatch.setenv("X", "nested_val")
        result = _substitute_dict({"outer": {"inner": "${X}"}})
        assert result == {"outer": {"inner": "nested_val"}}

    def test_list_of_strings(self, monkeypatch):
        monkeypatch.setenv("X", "item")
        result = _substitute_dict({"items": ["${X}", "static", "${X}"]})
        assert result == {"items": ["item", "static", "item"]}

    def test_non_string_values_untouched(self):
        result = _substitute_dict({"int": 42, "float": 3.14, "bool": True, "none": None})
        assert result == {"int": 42, "float": 3.14, "bool": True, "none": None}


# ============================================================================
# Constants
# ============================================================================


class TestEnvConstants:
    def test_constant_names_match_env_vars(self):
        assert ENV_LLM_BASE_URL == "LLM_BASE_URL"
        assert ENV_LLM_AUTH_ENABLED == "LLM_AUTH_ENABLED"
        assert ENV_LLM_API_KEY == "LLM_API_KEY"
        assert ENV_LLM_MODEL == "LLM_MODEL"


# ============================================================================
# Dataclass immutability & defaults
# ============================================================================


class TestSettingsDefaults:
    def test_settings_minimal_construction(self):
        s = Settings()
        assert s.rag.similarity_threshold == 0.75
        assert s.rag.top_k == 3
        assert s.memory.window_size == 10
        assert s.debug is False

    def test_settings_full_construction(self):
        s = Settings(
            rag=RAGSettings(similarity_threshold=0.5, top_k=1),
            memory=MemorySettings(window_size=5),
            debug=True,
        )
        assert s.rag.similarity_threshold == 0.5
        assert s.memory.window_size == 5
        assert s.debug is True
