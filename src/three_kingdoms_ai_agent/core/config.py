"""Configuration loader — reads YAML configs with ``${ENV_VAR}`` substitution.

LLM credentials are sourced exclusively from environment variables
(see :class:`LLMConfig`).  All other settings live in ``config/settings.yaml``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Shared YAML / env-var utilities
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _substitute_env(text: str) -> str:
    """Replace ``${ENV_VAR}`` patterns with ``os.environ`` values.

    Unknown variables are left as-is so missing configuration is obvious
    at runtime.
    """

    def _replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return _ENV_PATTERN.sub(_replacer, text)


def _substitute_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively apply :func:`_substitute_env` to every string value."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = _substitute_env(value)
        elif isinstance(value, dict):
            result[key] = _substitute_dict(value)
        elif isinstance(value, list):
            result[key] = [
                _substitute_env(item) if isinstance(item, str) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a single YAML file.

    Returns an empty dict when the file does not exist so callers don't
    need to guard every load.
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# LLM configuration — sourced from environment variables
# ---------------------------------------------------------------------------

# Environment variable names
ENV_LLM_BASE_URL = "LLM_BASE_URL"
ENV_LLM_AUTH_ENABLED = "LLM_AUTH_ENABLED"
ENV_LLM_API_KEY = "LLM_API_KEY"
ENV_LLM_MODEL = "LLM_MODEL"


@dataclass
class LLMConfig:
    """LLM provider configuration.

    All values are read from environment variables — nothing is hardcoded.
    Create via :meth:`from_env`.
    """

    base_url: str
    auth_enabled: bool
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Build an :class:`LLMConfig` from the four canonical environment variables.

        ::

            LLM_BASE_URL     → base_url
            LLM_AUTH_ENABLED → auth_enabled  ("true"/"false", default "true")
            LLM_API_KEY      → api_key
            LLM_MODEL        → model
        """
        return cls(
            base_url=os.environ.get(ENV_LLM_BASE_URL, ""),
            auth_enabled=os.environ.get(ENV_LLM_AUTH_ENABLED, "true").lower()
            == "true",
            api_key=os.environ.get(ENV_LLM_API_KEY, ""),
            model=os.environ.get(ENV_LLM_MODEL, ""),
        )

    def validate(self) -> list[str]:
        """Return a list of human-readable issues (empty ⇒ valid)."""
        issues: list[str] = []
        if not self.base_url:
            issues.append(f"{ENV_LLM_BASE_URL} is not set")
        if not self.model:
            issues.append(f"{ENV_LLM_MODEL} is not set")
        if self.auth_enabled and not self.api_key:
            issues.append(
                f"{ENV_LLM_AUTH_ENABLED}=true but {ENV_LLM_API_KEY} is not set"
            )
        return issues


# ---------------------------------------------------------------------------
# App settings — sourced from config/settings.yaml
# ---------------------------------------------------------------------------


@dataclass
class RAGSettings:
    """RAG / vector-search parameters."""

    similarity_threshold: float = 0.75
    top_k: int = 3


@dataclass
class MemorySettings:
    """Conversation memory parameters."""

    window_size: int = 10


@dataclass
class Settings:
    """Top-level application settings loaded from ``config/settings.yaml``."""

    rag: RAGSettings = field(default_factory=RAGSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    debug: bool = False


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_DIR = Path("config")


class ConfigLoader:
    """Loads application configuration from the ``config/`` directory.

    Usage::

        loader = ConfigLoader()
        llm_cfg = LLMConfig.from_env()       # env vars only
        settings = loader.load_settings()     # config/settings.yaml
        llm_opts = loader.load_llm_options()  # config/llm.yaml (operational)
    """

    def __init__(self, config_dir: str | Path = _DEFAULT_CONFIG_DIR) -> None:
        self._config_dir = Path(config_dir)

    # -- public API ----------------------------------------------------------

    def load_settings(self) -> Settings:
        """Load ``settings.yaml`` and return a typed :class:`Settings`."""
        raw = _load_yaml(self._config_dir / "settings.yaml")
        raw = _substitute_dict(raw)

        rag_raw = raw.get("rag", {})
        memory_raw = raw.get("memory", {})

        return Settings(
            rag=RAGSettings(
                similarity_threshold=float(
                    rag_raw.get("similarity_threshold", 0.75)
                ),
                top_k=int(rag_raw.get("top_k", 3)),
            ),
            memory=MemorySettings(
                window_size=int(memory_raw.get("window_size", 10)),
            ),
            debug=bool(raw.get("debug", False)),
        )

    def load_llm_options(self) -> dict[str, Any]:
        """Load ``llm.yaml`` (operational settings: timeout, max_retries, etc.).

        Credentials are **not** stored here — use :meth:`LLMConfig.from_env`.
        """
        raw = _load_yaml(self._config_dir / "llm.yaml")
        return _substitute_dict(raw)
