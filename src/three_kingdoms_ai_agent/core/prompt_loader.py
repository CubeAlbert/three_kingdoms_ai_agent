"""Prompt loader — reads agent and orchestrator prompts from YAML files.

All prompt files support ${ENV_VAR} substitution so sensitive values
(API keys, URLs) never appear in committed files.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentPrompt:
    """Structured prompt bundle for a sub-agent."""

    agent_id: str
    system_prompt: str
    sub_type_prompts: dict[str, str] = field(default_factory=dict)

    def get_prompt(self, sub_type: str) -> str | None:
        """Return the processing prompt for *sub_type*, or None."""
        return self.sub_type_prompts.get(sub_type)


@dataclass
class OrchestratorPrompt:
    """Prompt bundle for the orchestrator."""

    persona: str = ""
    chat_rules: str = ""
    integration_rules: str = ""


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------


class PromptLoader:
    """Loads prompt templates from a directory of YAML files.

    Directory layout expected::

        prompts/
          agents/
            recipe.yaml
            chat.yaml
            media.yaml
          orchestrator.yaml

    Each agent YAML::

        agent_id: recipe
        system_prompt: "你是一个……"
        sub_type_prompts:
          eat: "用户想吃东西时……"
          drink: "用户想喝东西时……"

    Orchestrator YAML::

        persona: "你是……"
        chat_rules: "规则……"
        integration_rules: "子Agent集成规则……"

    Every string value is scanned for ``${ENV_VAR}`` patterns and
    substituted at load time from ``os.environ``.
    """

    _ENV_PATTERN = re.compile(r"\$\{(\w+)\}")

    def __init__(self, prompts_dir: str | Path = "prompts") -> None:
        self._prompts_dir = Path(prompts_dir)
        self._cache: dict[str, AgentPrompt | OrchestratorPrompt] = {}

    # ---- public API -------------------------------------------------------

    def load_agent(self, agent_id: str) -> AgentPrompt:
        """Load prompts for *agent_id* from ``prompts/agents/{agent_id}.yaml``."""
        if agent_id in self._cache:
            cached = self._cache[agent_id]
            if isinstance(cached, AgentPrompt):
                return cached

        path = self._prompts_dir / "agents" / f"{agent_id}.yaml"
        raw = self._load_yaml(path)
        prompt = AgentPrompt(
            agent_id=raw.get("agent_id", agent_id),
            system_prompt=self._substitute(raw.get("system_prompt", "")),
            sub_type_prompts={
                k: self._substitute(v)
                for k, v in raw.get("sub_type_prompts", {}).items()
            },
        )
        self._cache[agent_id] = prompt
        return prompt

    def load_orchestrator(self) -> OrchestratorPrompt:
        """Load orchestrator prompts from ``prompts/orchestrator.yaml``."""
        cache_key = "__orchestrator__"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if isinstance(cached, OrchestratorPrompt):
                return cached

        path = self._prompts_dir / "orchestrator.yaml"
        raw = self._load_yaml(path)
        prompt = OrchestratorPrompt(
            persona=self._substitute(raw.get("persona", "")),
            chat_rules=self._substitute(raw.get("chat_rules", "")),
            integration_rules=self._substitute(
                raw.get("integration_rules", "")
            ),
        )
        self._cache[cache_key] = prompt
        return prompt

    def clear_cache(self) -> None:
        """Discard all cached prompts (useful for hot-reload during dev)."""
        self._cache.clear()

    # ---- internal helpers -------------------------------------------------

    def _load_yaml(self, path: Path) -> dict:
        """Read and parse a single YAML file.  Returns ``{}`` when the file
        does not exist so callers don't have to guard every load."""
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    @classmethod
    def _substitute(cls, text: str) -> str:
        """Replace ``${ENV_VAR}`` patterns with their environment values.

        Unknown variables are left untouched so missing config is obvious
        at runtime rather than silently swallowing tokens.
        """

        def _replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return cls._ENV_PATTERN.sub(_replacer, text)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def load_agent_prompt(
    agent_id: str, prompts_dir: str | Path = "prompts"
) -> AgentPrompt:
    """One-shot: load prompts for a single agent."""
    return PromptLoader(prompts_dir).load_agent(agent_id)


def load_orchestrator_prompt(
    prompts_dir: str | Path = "prompts",
) -> OrchestratorPrompt:
    """One-shot: load orchestrator prompts."""
    return PromptLoader(prompts_dir).load_orchestrator()
